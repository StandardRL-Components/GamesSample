
# Generated: 2025-08-27T15:00:52.033270
# Source Brief: brief_00860.md
# Brief Index: 860

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to push all blocks simultaneously."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Push all colored blocks onto their matching targets before the time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_TIME = 45  # in seconds
    MAX_STEPS = MAX_TIME * FPS

    GRID_COLS, GRID_ROWS = 12, 8
    CELL_SIZE = 40
    GRID_WIDTH_PX = GRID_COLS * CELL_SIZE
    GRID_HEIGHT_PX = GRID_ROWS * CELL_SIZE
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH_PX) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT_PX) // 2

    COLOR_BG = (28, 30, 40)
    COLOR_GRID = (40, 42, 54)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_WARN = (255, 100, 100)
    COLOR_WIN = (180, 255, 180)
    COLOR_LOSE = (255, 180, 180)

    BLOCK_COLORS = [
        (255, 80, 80),   # Red
        (80, 200, 255),  # Blue
        (80, 255, 120),  # Green
        (255, 220, 80),  # Yellow
    ]
    TARGET_COLORS = [
        (c[0] // 3, c[1] // 3, c[2] // 3) for c in BLOCK_COLORS
    ]
    
    NUM_BLOCKS = 4
    SHUFFLE_MOVES = 20
    ANIMATION_SPEED = 16  # pixels per frame

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 50)
        
        self.blocks = []
        self.targets = []
        self.particles = []
        self.is_pushing = False
        self.win = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves = 0
        self.game_over = False
        self.win = False
        self.time_remaining = self.MAX_TIME
        self.is_pushing = False
        self.particles.clear()
        
        self._generate_puzzle()
        self.blocks_on_target_count = self._count_blocks_on_target()
        
        return self._get_observation(), self._get_info()
    
    def _generate_puzzle(self):
        self.blocks.clear()
        self.targets.clear()

        all_coords = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        
        # Ensure there are enough coordinates
        if len(all_coords) < self.NUM_BLOCKS:
            raise ValueError("Grid is too small for the number of blocks.")
            
        target_coords_indices = self.np_random.choice(len(all_coords), size=self.NUM_BLOCKS, replace=False)
        target_coords = [all_coords[i] for i in target_coords_indices]

        for i in range(self.NUM_BLOCKS):
            tx, ty = target_coords[i]
            self.targets.append({'pos': (tx, ty), 'color': self.TARGET_COLORS[i]})
            
            block = {
                'id': i,
                'pos': [tx, ty],
                'pixel_pos': self._grid_to_pixel(tx, ty),
                'color': self.BLOCK_COLORS[i],
                'target_pos': (tx, ty)
            }
            self.blocks.append(block)

        # Shuffle the puzzle with random moves to ensure solvability
        for _ in range(self.SHUFFLE_MOVES):
            move = self.np_random.integers(1, 5)
            self._apply_push_logic(move, animate=False)
        
        # If puzzle is solved after shuffling (rare), shuffle again
        if self._count_blocks_on_target() == self.NUM_BLOCKS:
            self.reset() # Easiest way to re-generate
            return

        # Set initial pixel positions after shuffling
        for block in self.blocks:
            block['pixel_pos'] = self._grid_to_pixel(*block['pos'])

    def step(self, action):
        movement = action[0]
        
        frame_reward = -0.01  # Small penalty for time passing
        
        if self.game_over:
            movement = 0
        else:
            self.time_remaining -= 1 / self.FPS
        
        if movement != 0 and not self.is_pushing:
            if self._apply_push_logic(movement, animate=True):
                self.moves += 1
                frame_reward -= 0.1 # Cost per move
                # sfx: push
        
        if self.is_pushing:
            animation_finished = self._update_push_animation()
            if animation_finished:
                self.is_pushing = False
                new_on_target = self._count_blocks_on_target()
                newly_placed = new_on_target - self.blocks_on_target_count
                if newly_placed > 0:
                    frame_reward += newly_placed * 1.0
                    # sfx: goal
                    for block in self.blocks:
                        if self._is_block_on_target(block):
                            self._create_particles(block['pixel_pos'], block['color'])
                self.blocks_on_target_count = new_on_target

        self._update_particles()
        
        self.score += frame_reward
        terminated = self._check_termination(frame_reward)
        
        return (
            self._get_observation(),
            frame_reward,
            terminated,
            False,
            self._get_info()
        )

    def _apply_push_logic(self, movement, animate=True):
        if movement == 1: dx, dy, sort_key, reverse = 0, -1, 1, False # Up
        elif movement == 2: dx, dy, sort_key, reverse = 0, 1, 1, True  # Down
        elif movement == 3: dx, dy, sort_key, reverse = -1, 0, 0, False # Left
        elif movement == 4: dx, dy, sort_key, reverse = 1, 0, 0, True  # Right
        else: return False

        sorted_blocks = sorted(self.blocks, key=lambda b: b['pos'][sort_key], reverse=reverse)
        
        occupied = {tuple(b['pos']) for b in self.blocks}
        new_positions = {}
        has_moved = False

        for block in sorted_blocks:
            current_pos = tuple(block['pos'])
            final_pos = list(current_pos)
            
            while True:
                next_pos = (final_pos[0] + dx, final_pos[1] + dy)
                if not (0 <= next_pos[0] < self.GRID_COLS and 0 <= next_pos[1] < self.GRID_ROWS):
                    break # Hit wall
                if next_pos in occupied:
                    break # Hit other block
                final_pos = list(next_pos)

            if tuple(final_pos) != current_pos:
                has_moved = True
                occupied.remove(current_pos)
                occupied.add(tuple(final_pos))
            new_positions[block['id']] = tuple(final_pos)

        if has_moved:
            for block in self.blocks:
                block['pos'] = list(new_positions[block['id']])
                if animate:
                    block['target_pixel_pos'] = self._grid_to_pixel(*block['pos'])
            if animate:
                self.is_pushing = True
        return has_moved

    def _update_push_animation(self):
        all_finished = True
        for block in self.blocks:
            px, py = block['pixel_pos']
            tx, ty = block['target_pixel_pos']
            
            dx, dy = tx - px, ty - py
            dist = math.hypot(dx, dy)

            if dist < self.ANIMATION_SPEED:
                block['pixel_pos'] = [tx, ty]
            else:
                all_finished = False
                block['pixel_pos'][0] += (dx / dist) * self.ANIMATION_SPEED
                block['pixel_pos'][1] += (dy / dist) * self.ANIMATION_SPEED
        return all_finished

    def _check_termination(self, frame_reward):
        if self.game_over:
            return True

        if self.blocks_on_target_count == self.NUM_BLOCKS:
            self.game_over = True
            self.win = True
            self.score += 50 - frame_reward # Overwrite frame reward with final bonus
            # sfx: win
            return True

        if self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            self.score += -50 - frame_reward # Overwrite with final penalty
            # sfx: lose
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
        for x in range(self.GRID_COLS + 1):
            px = self.GRID_OFFSET_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_OFFSET_Y), (px, self.GRID_OFFSET_Y + self.GRID_HEIGHT_PX))
        for y in range(self.GRID_ROWS + 1):
            py = self.GRID_OFFSET_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, py), (self.GRID_OFFSET_X + self.GRID_WIDTH_PX, py))

        # Draw targets
        radius = self.CELL_SIZE // 3
        for target in self.targets:
            px, py = self._grid_to_pixel(*target['pos'])
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), radius, target['color'])
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), radius, target['color'])

        # Draw blocks
        size = self.CELL_SIZE * 0.8
        corner_radius = int(size * 0.2)
        for block in self.blocks:
            px, py = block['pixel_pos']
            rect = pygame.Rect(px - size / 2, py - size / 2, size, size)
            pygame.draw.rect(self.screen, block['color'], rect, border_radius=corner_radius)

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (p['pos'][0] - p['size'], p['pos'][1] - p['size']))

    def _render_ui(self):
        # Time
        time_text = f"TIME: {max(0, self.time_remaining):.1f}"
        time_color = self.COLOR_TEXT if self.time_remaining > 10 else self.COLOR_TEXT_WARN
        time_surf = self.font_ui.render(time_text, True, time_color)
        self.screen.blit(time_surf, (10, 10))

        # Score
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))
        
        # Moves
        moves_text = f"MOVES: {self.moves}"
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(moves_surf, (10, 40))

        # Game Over Message
        if self.game_over:
            msg = "PUZZLE SOLVED!" if self.win else "TIME'S UP!"
            color = self.COLOR_WIN if self.win else self.COLOR_LOSE
            msg_surf = self.font_msg.render(msg, True, color)
            pos = (self.SCREEN_WIDTH / 2 - msg_surf.get_width() / 2, self.SCREEN_HEIGHT / 2 - msg_surf.get_height() / 2)
            self.screen.blit(msg_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves": self.moves,
            "time_remaining": self.time_remaining,
            "blocks_on_target": self.blocks_on_target_count,
        }

    # --- Helper Functions ---
    def _grid_to_pixel(self, x, y):
        px = self.GRID_OFFSET_X + (x + 0.5) * self.CELL_SIZE
        py = self.GRID_OFFSET_Y + (y + 0.5) * self.CELL_SIZE
        return [px, py]

    def _count_blocks_on_target(self):
        count = 0
        for block in self.blocks:
            if self._is_block_on_target(block):
                count += 1
        return count

    def _is_block_on_target(self, block):
        return self.targets[block['id']]['pos'] == tuple(block['pos'])
        
    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'life': self.np_random.uniform(15, 30),
                'max_life': 30,
                'size': self.np_random.integers(2, 5),
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Pusher")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op by default
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_q:
                    running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        # Construct the action
        action = [movement, 0, 0] # Space and shift are not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Moves: {info['moves']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    pygame.quit()