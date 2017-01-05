#include "asr.h"

#define F_MAX 2.0
#define F_MIN -2.0

void read_model2(RNN &model, FILE *model_file)
{
	/*Read model from file*/

	int num_layers;
	fscanf(model_file, "%d\n", &num_layers);
	model.num_layers = num_layers;
	for (int i = 0; i < num_layers; i++)
	{
		//printf("Obtaining layer %d information\n", i);
		char layer_type[10];

		fscanf(model_file, "%s\n", &layer_type);
		string cpp_layer_type(layer_type);
		//std::cout << "Layer Type " << cpp_layer_type << endl;
		model.layer_type.push_back(cpp_layer_type);
		if (cpp_layer_type == "D")
		{
			int num_rows;
			int num_cols;
			fscanf(model_file, "%d", &num_rows);
			fscanf(model_file, "%d\n", &num_cols);
			//printf("Number of rows and cols %d %d \n", num_rows, num_cols);
			struct weight_matrix *Wi = alloc_weight_matrix(num_rows, num_cols);
			for (int r = 0; r < num_cols; r++)
				for (int c = 0; c < num_rows; c++)
				{
					if ((r == num_cols - 1) && (c == num_rows - 2)) break;
					fscanf(model_file, "%f", &(Wi->MAT[num_cols*c + r]));
				}

			Wi->MAT[num_cols*(num_rows - 2) + (num_cols - 1)] = 0.0;
			Wi->MAT[num_cols*(num_rows - 1) + (num_cols - 1)] = 0.0;

			model.W.push_back(Wi);
			int len;
			fscanf(model_file, "%d", &len);
			struct bias_vector *bi = alloc_bias_vector(len);
			for (int k = 0; k < len - 2; k++)
				fscanf(model_file, "%f", &(bi->ARRAY[k]));
			bi->ARRAY[len - 2] = 0.0;
			bi->ARRAY[len - 1] = 0.0;
			model.b.push_back(bi);
		}
		else if (cpp_layer_type == "RU")
		{
			int len;

			fscanf(model_file, "%d\n", &len);
			struct bias_vector *init = alloc_bias_vector(len);
			for (int k = 0; k < len; k++)
				fscanf(model_file, "%f", &(init->ARRAY[k]));
			model.h_init.push_back(init);


			for (int num_mat = 0; num_mat < 2; num_mat++)
			{
				int num_rows;
				int num_cols;

				fscanf(model_file, "%d", &num_rows);
				fscanf(model_file, "%d\n", &num_cols);
				//printf("Number of rows and cols %d %d \n", num_rows, num_cols);


				struct weight_matrix *Wi = alloc_weight_matrix(num_rows, num_cols);
				for (int r = 0; r < num_cols; r++)
					for (int c = 0; c < num_rows; c++)
					{
						//printf("Scanning element (%d,%d)\n", r, c);
						fscanf(model_file, "%f", &(Wi->MAT[num_cols*c + r]));

					}
				model.W.push_back(Wi);
				//printf("Completed scanning matrix %d\n", num_mat);
			}

			fscanf(model_file, "%d\n", &len);
			struct bias_vector *bi = alloc_bias_vector(len);
			for (int k = 0; k < len; k++)
				fscanf(model_file, "%f", &(bi->ARRAY[k]));
			model.b.push_back(bi);

		}
		else if (cpp_layer_type == "RB")
		{

			for (int num_mat = 0; num_mat < 4; num_mat++)
			{
				int num_rows;
				int num_cols;
				fscanf(model_file, "%d", &num_rows);
				fscanf(model_file, "%d\n", &num_cols);
				printf("Number of rows and cols %d %d \n", num_rows, num_cols);
				struct weight_matrix *Wi = alloc_weight_matrix(num_rows, num_cols);
				for (int r = 0; r < num_cols; r++)
					for (int c = 0; c < num_rows; c++)
					{
						if ((r == num_cols - 1) && (c == num_rows - 2)) break;
						fscanf(model_file, "%f", &(Wi->MAT[num_cols*c + r]));
					}

				Wi->MAT[num_cols*(num_rows - 2) + (num_cols - 1)] = 0.0;
				Wi->MAT[num_cols*(num_rows - 1) + (num_cols - 1)] = 0.0;
				//printf("%f\n", Wi->MAT[10]);
				model.W.push_back(Wi);
				printf("Scanned matrix %d\n", num_mat);

			}
			//printf("Completed Matrix Output");

			for (int num_vec = 0; num_vec < 2; num_vec++)
			{

				int len;
				fscanf(model_file, "%d\n", &len);
				struct bias_vector *bi = alloc_bias_vector(len);
				for (int k = 0; k < len - 2; k++)
					fscanf(model_file, "%f", &(bi->ARRAY[k]));
				bi->ARRAY[len - 2] = 0.0;
				bi->ARRAY[len - 1] = 0.0;
				model.b.push_back(bi);
				//printf("Completed scanning vector %d\n", num_vec);
			}

		}


	}

	//printf("Number of layers %d\n", model.num_layers);
	fclose(model_file);
	return;

}

float rand_float()
{
	return F_MIN + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (F_MAX - F_MIN)));
}

void rand_init_bias(bias_vector *b)
{
	int len = b->len;
	float randval;
	for (int i = 0; i < len; i++)
	{
		randval = rand_float();
		b->ARRAY[i] = randval;
	}

}

void rand_init_weight(weight_matrix *W)
{
	int len = (W->num_rows)*(W->num_cols);
	float randval;
	for (int i = 0; i < len; i++)
	{
		randval = rand_float();
		W->MAT[i] = randval;
	}

}



void generate_model(RNN &model, int hidden_size)
{
	printf("Generating a random CTC model with hidden layer size %d\n", hidden_size);
	model.num_layers = 3;

	model.layer_type.push_back("RB");

	weight_matrix * w = alloc_weight_matrix(hidden_size, 39);
	rand_init_weight(w);
	model.W.push_back(w);

	w = alloc_weight_matrix(hidden_size, 39);
	rand_init_weight(w);
	model.W.push_back(w);

	w = alloc_weight_matrix(hidden_size, hidden_size);
	rand_init_weight(w);
	model.W.push_back(w);

	w = alloc_weight_matrix(hidden_size, hidden_size);
	rand_init_weight(w);
	model.W.push_back(w);

	bias_vector * b = alloc_bias_vector(hidden_size);
	rand_init_bias(b);
	model.b.push_back(b);

	b = alloc_bias_vector(hidden_size);
	rand_init_bias(b);
	model.b.push_back(b);

	model.layer_type.push_back("D");

	w = alloc_weight_matrix(hidden_size, hidden_size);
	rand_init_weight(w);
	model.W.push_back(w);

	b = alloc_bias_vector(hidden_size);
	rand_init_bias(b);
	model.b.push_back(b);


	model.layer_type.push_back("D");

	w = alloc_weight_matrix(30, hidden_size);
	rand_init_weight(w);
	model.W.push_back(w);

	b = alloc_bias_vector(30);
	rand_init_bias(b);
	model.b.push_back(b);

	printf("Model Generated\n");
}