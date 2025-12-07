
# Generated: 2025-08-28T02:32:29.175237
# Source Brief: brief_04478.md
# Brief Index: 4478

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Use arrow keys to move the selector. Hold Shift and press an arrow key to swap the pixel under the selector with an adjacent one."
    )

    game_description = (
        "A retro-styled puzzle game. Strategically swap pixels on the grid to recreate the target image before you run out of moves."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = (12, 8)
        self.CELL_SIZE = 36
        self.MAX_MOVES = 30
        self.MAX_STEPS = 1000
        
        # Grid positioning
        self.GRID_WIDTH = self.GRID_SIZE[0] * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE[1] * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = 60

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_SELECTOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.TARGET_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 80, 255),   # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
        ]

        # Initialize state variables
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.cursor_pos = (0, 0)
        self.pixels = []
        self.pixel_map = {}
        self.moves_left = 0
        self.particles = []
        self.target_pattern = []
        self.puzzle_pixel_count = 0
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = (self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2)
        self.particles.clear()
        
        # Define the target pattern (e.g., a simple shape)
        self.target_pattern = [
            {'coord': (4, 2), 'color_idx': 3}, {'coord': (7, 2), 'color_idx': 3},
            {'coord': (3, 4), 'color_idx': 3}, {'coord': (4, 4), 'color_idx': 3},
            {'coord': (5, 4), 'color_idx': 3}, {'coord': (6, 4), 'color_idx': 3},
            {'coord': (7, 4), 'color_idx': 3}, {'coord': (8, 4), 'color_idx': 3},
        ]
        self.puzzle_pixel_count = len(self.target_pattern)

        # Create pixels and shuffle their starting positions
        self.pixels.clear()
        available_coords = [(x, y) for x in range(self.GRID_SIZE[0]) for y in range(self.GRID_SIZE[1])]
        self.np_random.shuffle(available_coords)

        for i, p_info in enumerate(self.target_pattern):
            start_coord = available_coords.pop()
            self.pixels.append({
                'id': i,
                'target_coord': p_info['coord'],
                'current_coord': start_coord,
                'color_idx': p_info['color_idx']
            })

        self._update_pixel_map()
        
        return self._get_observation(), self._get_info()

    def _update_pixel_map(self):
        self.pixel_map.clear()
        for i, pixel in enumerate(self.pixels):
            self.pixel_map[pixel['current_coord']] = i

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = 0
        self.steps += 1
        
        is_swap_action = shift_held and movement > 0 and self.moves_left > 0

        if is_swap_action:
            self.moves_left -= 1
            reward += self._perform_swap(movement)
        elif movement > 0:
            self._move_cursor(movement)

        # Update score and check for termination
        self.score += reward
        terminated = self._check_termination()
        self.game_over = terminated

        if terminated and self._is_solved():
            # Goal-oriented reward for solving the puzzle
            bonus = 50 * (self.moves_left / self.MAX_MOVES)
            reward += bonus
            self.score += bonus
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _perform_swap(self, movement):
        reward = 0
        
        # Pixel A is under the cursor
        coord_a = self.cursor_pos
        dx, dy = self._get_move_vector(movement)
        coord_b = ((coord_a[0] + dx) % self.GRID_SIZE[0], (coord_a[1] + dy) % self.GRID_SIZE[1])

        idx_a = self.pixel_map.get(coord_a)
        idx_b = self.pixel_map.get(coord_b)

        # Calculate rewards based on distance change
        if idx_a is not None:
            pixel_a = self.pixels[idx_a]
            reward += self._calculate_distance_reward(pixel_a, coord_b)
            if coord_b == pixel_a['target_coord']:
                reward += 1.0
                self._spawn_particles(coord_b, pixel_a['color_idx'])
                # sfx: correct_place.wav

        if idx_b is not None:
            pixel_b = self.pixels[idx_b]
            reward += self._calculate_distance_reward(pixel_b, coord_a)
            if coord_a == pixel_b['target_coord']:
                reward += 1.0
                self._spawn_particles(coord_a, pixel_b['color_idx'])
                # sfx: correct_place.wav

        # Perform the swap in the state
        if idx_a is not None: self.pixels[idx_a]['current_coord'] = coord_b
        if idx_b is not None: self.pixels[idx_b]['current_coord'] = coord_a
        
        self._update_pixel_map()
        # sfx: swap.wav
        return reward

    def _calculate_distance_reward(self, pixel, new_coord):
        def dist(c1, c2):
            return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])

        old_dist = dist(pixel['current_coord'], pixel['target_coord'])
        new_dist = dist(new_coord, pixel['target_coord'])

        if new_dist < old_dist: return 0.1
        if new_dist > old_dist: return -0.1
        return 0

    def _move_cursor(self, movement):
        dx, dy = self._get_move_vector(movement)
        self.cursor_pos = (
            (self.cursor_pos[0] + dx) % self.GRID_SIZE[0],
            (self.cursor_pos[1] + dy) % self.GRID_SIZE[1]
        )
        # sfx: cursor_move.wav

    def _get_move_vector(self, movement):
        if movement == 1: return 0, -1  # Up
        if movement == 2: return 0, 1   # Down
        if movement == 3: return -1, 0  # Left
        if movement == 4: return 1, 0   # Right
        return 0, 0
    
    def _is_solved(self):
        return all(p['current_coord'] == p['target_coord'] for p in self.pixels)

    def _check_termination(self):
        if self._is_solved():
            return True
        if self.moves_left <= 0:
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
        # Draw grid lines
        for x in range(self.GRID_SIZE[0] + 1):
            start_pos = (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET)
            end_pos = (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        for y in range(self.GRID_SIZE[1] + 1):
            start_pos = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + y * self.CELL_SIZE)
            end_pos = (self.GRID_X_OFFSET + self.GRID_WIDTH, self.GRID_Y_OFFSET + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

        # Draw pixels
        for pixel in self.pixels:
            gx, gy = pixel['current_coord']
            px = self.GRID_X_OFFSET + gx * self.CELL_SIZE
            py = self.GRID_Y_OFFSET + gy * self.CELL_SIZE
            
            base_color = self.TARGET_COLORS[pixel['color_idx']]
            
            is_correct = pixel['current_coord'] == pixel['target_coord']
            if is_correct:
                # Pulse effect for correctly placed pixels
                pulse = (math.sin(self.steps * 0.2) + 1) / 2  # Varies between 0 and 1
                color = tuple(int(c * (0.8 + pulse * 0.2)) for c in base_color)
            else:
                # Muted color for incorrectly placed pixels
                color = tuple(int(c * 0.5) for c in base_color)
            
            pygame.draw.rect(self.screen, color, (px + 3, py + 3, self.CELL_SIZE - 6, self.CELL_SIZE - 6), border_radius=4)
        
        # Draw cursor
        cx, cy = self.cursor_pos
        px = self.GRID_X_OFFSET + cx * self.CELL_SIZE
        py = self.GRID_Y_OFFSET + cy * self.CELL_SIZE
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, (px, py, self.CELL_SIZE, self.CELL_SIZE), 3, border_radius=6)

        # Update and draw particles
        self._update_and_draw_particles()

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
                alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
                color = p['color'] + (alpha,)
                size = int(p['size'] * (p['life'] / p['max_life']))
                if size > 0:
                    rect = pygame.Rect(int(p['pos'][0] - size/2), int(p['pos'][1] - size/2), size, size)
                    pygame.gfxdraw.box(self.screen, rect, color)
        self.particles = active_particles

    def _spawn_particles(self, grid_coord, color_idx):
        px = self.GRID_X_OFFSET + grid_coord[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        py = self.GRID_Y_OFFSET + grid_coord[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        color = self.TARGET_COLORS[color_idx]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            life = random.randint(15, 30)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'size': random.randint(4, 8)
            })
    
    def _render_ui(self):
        # Score and Moves
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))
        
        moves_text = self.font_large.render(f"MOVES: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 15))

        # Target Image Preview
        preview_size = 12
        preview_x = self.WIDTH // 2 - (self.GRID_SIZE[0] * preview_size) // 2
        preview_y = 10
        pygame.draw.rect(self.screen, self.COLOR_GRID, (preview_x - 2, preview_y - 2, self.GRID_SIZE[0] * preview_size + 4, self.GRID_SIZE[1] * preview_size + 4), 1, 2)
        for p_info in self.target_pattern:
            gx, gy = p_info['coord']
            color = self.TARGET_COLORS[p_info['color_idx']]
            pygame.draw.rect(self.screen, color, (preview_x + gx * preview_size, preview_y + gy * preview_size, preview_size - 1, preview_size - 1))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "PUZZLE SOLVED!" if self._is_solved() else "OUT OF MOVES"
            end_text = self.font_large.render(message, True, (255, 255, 100))
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "correct_pixels": sum(1 for p in self.pixels if p['current_coord'] == p['target_coord'])
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
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pixel Swap Puzzle")
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("      PIXEL SWAP PUZZLE")
    print("="*30)
    print(env.game_description)
    print("\nCONTROLS:")
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # Map keyboard inputs to the action space
        keys = pygame.key.get_pressed()
        
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Only step if an action is taken, because auto_advance is False
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Moves Left: {info['moves_left']}")

            if terminated:
                print("\nGAME OVER!")
                print(f"Final Score: {info['score']:.2f}")
                obs, info = env.reset() # Auto-reset
                total_reward = 0
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("\n--- RESETTING GAME ---")
                obs, info = env.reset()
                total_reward = 0

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate for human play

    env.close()