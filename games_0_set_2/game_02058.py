
# Generated: 2025-08-27T19:07:27.840007
# Source Brief: brief_02058.md
# Brief Index: 2058

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move selected car. Space/Shift to cycle selection."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Untangle the traffic jam. Move the green car to the exit on the right."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 6, 6
        self.CELL_SIZE = 60
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        self.MAX_MOVES = 20
        self.MAX_STEPS = 1000 # Safety break

        # --- Colors ---
        self.COLOR_BG = (34, 38, 54)
        self.COLOR_GRID = (50, 58, 79)
        self.COLOR_EXIT = (40, 167, 69, 100) # RGBA for transparency
        self.COLOR_TARGET_CAR = (40, 167, 69)
        self.COLOR_CAR_PALETTE = [
            (220, 53, 69), (255, 193, 7), (0, 123, 255), 
            (108, 117, 125), (253, 126, 20), (111, 66, 193)
        ]
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_SELECTION = (255, 255, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # --- Game State (initialized in reset) ---
        self.cars = []
        self.target_car_idx = -1
        self.selected_car_idx = -1
        self.exit_row = 0
        self.steps = 0
        self.moves_taken = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # Initialize state variables by calling reset
        self.reset()

        # Final validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.moves_taken = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over or self.steps >= self.MAX_STEPS:
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # --- Action: Cycle Selection ---
        if space_held and not self.prev_space_held:
            self.selected_car_idx = (self.selected_car_idx + 1) % len(self.cars)
            # Sound placeholder: pygame.mixer.Sound('select.wav').play()
        if shift_held and not self.prev_shift_held:
            self.selected_car_idx = (self.selected_car_idx - 1 + len(self.cars)) % len(self.cars)
            # Sound placeholder: pygame.mixer.Sound('select.wav').play()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Action: Move Car ---
        if movement > 0 and self.selected_car_idx != -1:
            car = self.cars[self.selected_car_idx]
            original_pos = (car['x'], car['y'])
            
            dx, dy = 0, 0
            if movement == 1 and car['is_vertical']: dy = -1  # Up
            elif movement == 2 and car['is_vertical']: dy = 1  # Down
            elif movement == 3 and not car['is_vertical']: dx = -1 # Left
            elif movement == 4 and not car['is_vertical']: dx = 1  # Right

            if (dx != 0 or dy != 0) and self._is_move_legal(self.selected_car_idx, dx, dy):
                car['x'] += dx
                car['y'] += dy
                self.moves_taken += 1
                self.score -= 0.1 # Cost for making a move
                # Sound placeholder: pygame.mixer.Sound('move.wav').play()
                
                # Reward for moving target car closer to exit
                if self.selected_car_idx == self.target_car_idx:
                    if dx > 0: # Moving right towards exit
                        self.score += 1.0
                        reward += 1.0
                    elif dx < 0: # Moving left away from exit
                        self.score -= 1.0
                        reward -= 1.0
            else:
                # Sound placeholder: pygame.mixer.Sound('illegal_move.wav').play()
                pass # Illegal move, no change

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            target_car = self.cars[self.target_car_idx]
            if target_car['x'] >= self.GRID_WIDTH - (target_car['length'] - 1): # Win
                self.score += 100
                reward += 100
            else: # Loss (max moves or no moves left)
                self.score -= 100
                reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_puzzle(self):
        max_attempts = 100
        for _ in range(max_attempts):
            if self._try_generate_puzzle():
                return
        # Failsafe: if generation fails, create a trivial puzzle
        self._create_failsafe_puzzle()

    def _try_generate_puzzle(self):
        self.cars = []
        occupied_cells = set()
        
        # 1. Place target car at solved position
        self.exit_row = self.np_random.integers(1, self.GRID_HEIGHT - 1)
        target_car = {
            'x': self.GRID_WIDTH - 2, 'y': self.exit_row, 'length': 2, 
            'is_vertical': False, 'color': self.COLOR_TARGET_CAR, 'is_target': True
        }
        self.cars.append(target_car)
        self.target_car_idx = 0
        for i in range(target_car['length']):
            occupied_cells.add((target_car['x'] + i, target_car['y']))

        # 2. Place other cars
        num_other_cars = self.np_random.integers(8, 13)
        for _ in range(num_other_cars):
            for _ in range(50): # Placement attempts
                car = {
                    'length': self.np_random.integers(2, 4),
                    'is_vertical': self.np_random.choice([True, False]),
                    'color': self.np_random.choice(self.COLOR_CAR_PALETTE),
                    'is_target': False
                }
                if car['is_vertical']:
                    car['x'] = self.np_random.integers(0, self.GRID_WIDTH)
                    car['y'] = self.np_random.integers(0, self.GRID_HEIGHT - car['length'] + 1)
                else:
                    car['x'] = self.np_random.integers(0, self.GRID_WIDTH - car['length'] + 1)
                    car['y'] = self.np_random.integers(0, self.GRID_HEIGHT)
                
                new_cells = self._get_car_cells(car)
                if not any(cell in occupied_cells for cell in new_cells):
                    self.cars.append(car)
                    occupied_cells.update(new_cells)
                    break
        
        # 3. Shuffle the board with random moves
        shuffle_moves = 50
        for _ in range(shuffle_moves):
            car_idx = self.np_random.integers(0, len(self.cars))
            car = self.cars[car_idx]
            move = self.np_random.choice([-1, 1])
            dx, dy = (0, move) if car['is_vertical'] else (move, 0)
            if self._is_move_legal(car_idx, dx, dy):
                car['x'] += dx
                car['y'] += dy
        
        self.selected_car_idx = self.target_car_idx

        # 4. Validate puzzle
        if self.cars[self.target_car_idx]['x'] >= self.GRID_WIDTH - 1: return False # Already solved
        if not self._has_any_legal_move(): return False # No moves left
        
        return True

    def _create_failsafe_puzzle(self):
        self.cars = []
        self.exit_row = 3
        # Target Car
        self.cars.append({'x': 1, 'y': self.exit_row, 'length': 2, 'is_vertical': False, 'color': self.COLOR_TARGET_CAR, 'is_target': True})
        # Blocker
        self.cars.append({'x': 3, 'y': self.exit_row, 'length': 2, 'is_vertical': True, 'color': self.COLOR_CAR_PALETTE[0], 'is_target': False})
        self.target_car_idx = 0
        self.selected_car_idx = 0

    def _is_move_legal(self, car_idx, dx, dy):
        car_to_move = self.cars[car_idx]
        new_x, new_y = car_to_move['x'] + dx, car_to_move['y'] + dy

        # Check grid boundaries
        if car_to_move['is_vertical']:
            if not (0 <= new_x < self.GRID_WIDTH and 0 <= new_y and new_y + car_to_move['length'] <= self.GRID_HEIGHT):
                return False
        else:
            if not (0 <= new_y < self.GRID_HEIGHT and 0 <= new_x and new_x + car_to_move['length'] <= self.GRID_WIDTH):
                return False
        
        # Check for collisions with other cars
        car_cells = self._get_car_cells({'x': new_x, 'y': new_y, 'length': car_to_move['length'], 'is_vertical': car_to_move['is_vertical']})
        for i, other_car in enumerate(self.cars):
            if i == car_idx:
                continue
            other_car_cells = self._get_car_cells(other_car)
            if any(cell in other_car_cells for cell in car_cells):
                return False
        return True

    def _get_car_cells(self, car):
        cells = []
        for i in range(car['length']):
            if car['is_vertical']:
                cells.append((car['x'], car['y'] + i))
            else:
                cells.append((car['x'] + i, car['y']))
        return cells

    def _has_any_legal_move(self):
        for i, car in enumerate(self.cars):
            if car['is_vertical']:
                if self._is_move_legal(i, 0, 1) or self._is_move_legal(i, 0, -1):
                    return True
            else:
                if self._is_move_legal(i, 1, 0) or self._is_move_legal(i, -1, 0):
                    return True
        return False

    def _check_termination(self):
        # Win condition
        target_car = self.cars[self.target_car_idx]
        if target_car['x'] >= self.GRID_WIDTH - (target_car['length'] - 1):
            return True
        
        # Loss conditions
        if self.moves_taken >= self.MAX_MOVES:
            return True
        if not self._has_any_legal_move():
            return True
        if self.steps >= self.MAX_STEPS:
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, 
                                self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)

        # Draw exit zone
        exit_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        exit_surf.fill(self.COLOR_EXIT)
        self.screen.blit(exit_surf, (self.GRID_X_OFFSET + (self.GRID_WIDTH - 1) * self.CELL_SIZE, 
                                     self.GRID_Y_OFFSET + self.exit_row * self.CELL_SIZE))

        # Draw cars
        for i, car in enumerate(self.cars):
            w = car['length'] * self.CELL_SIZE if not car['is_vertical'] else self.CELL_SIZE
            h = car['length'] * self.CELL_SIZE if car['is_vertical'] else self.CELL_SIZE
            x = self.GRID_X_OFFSET + car['x'] * self.CELL_SIZE
            y = self.GRID_Y_OFFSET + car['y'] * self.CELL_SIZE
            
            car_rect = pygame.Rect(x, y, w, h)
            
            # Draw car with a border
            pygame.draw.rect(self.screen, car['color'], car_rect.inflate(-8, -8))
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in car['color']), car_rect, 4)

            # Draw selection highlight
            if i == self.selected_car_idx:
                pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 # 0 to 1
                width = int(2 + pulse * 3)
                pygame.draw.rect(self.screen, self.COLOR_SELECTION, car_rect, width)
            
            # Draw target car indicator
            if car['is_target']:
                pygame.gfxdraw.filled_trigon(self.screen, 
                                             car_rect.right - 15, car_rect.centery - 8,
                                             car_rect.right - 15, car_rect.centery + 8,
                                             car_rect.right - 5, car_rect.centery,
                                             (255, 255, 255))


    def _render_ui(self):
        # Moves display
        moves_text = f"Moves: {self.moves_taken}/{self.MAX_MOVES}"
        text_surf = self.font_small.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (20, 20))
        
        # Score display
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_surf, score_rect)

        # Game over message
        if self.game_over:
            target_car = self.cars[self.target_car_idx]
            is_win = target_car['x'] >= self.GRID_WIDTH - (target_car['length'] - 1)
            msg = "PUZZLE SOLVED!" if is_win else "GAME OVER"
            color = self.COLOR_TARGET_CAR if is_win else (220, 53, 69)
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg_surf = self.font_large.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_taken": self.moves_taken,
            "game_over": self.game_over
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headlessly

    env = GameEnv(render_mode="rgb_array")
    
    # --- Test with a random agent ---
    obs, info = env.reset()
    print("Initial state:")
    print(info)
    
    terminated = False
    total_reward = 0
    for i in range(200): # Run for max 200 steps
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i % 10 == 0) or terminated:
            print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Total Reward={total_reward:.2f}, Terminated={terminated}, Info={info}")
        if terminated:
            break
            
    print("\n--- Episode Finished ---")
    print(f"Final Info: {info}")
    print(f"Total Reward: {total_reward}")

    env.close()