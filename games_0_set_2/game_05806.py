
# Generated: 2025-08-28T06:10:49.154291
# Source Brief: brief_05806.md
# Brief Index: 5806

        
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
        "Controls: Use arrows to move the selector. Hold Space and press an arrow key to push a block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Push colored blocks onto their matching targets. "
        "Plan your moves carefully, as you only have a limited number of pushes!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 8
    CELL_SIZE = 40
    GRID_X_OFFSET = (WIDTH - GRID_SIZE * CELL_SIZE) // 2
    GRID_Y_OFFSET = (HEIGHT - GRID_SIZE * CELL_SIZE) // 2 + 20
    MAX_MOVES = 20
    PARTICLE_LIFESPAN = 30

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (50, 60, 70)
    COLOR_SELECTOR = (255, 255, 0)
    BLOCK_COLORS = [
        (220, 70, 70),   # Red
        (70, 220, 70),   # Green
        (70, 120, 255),  # Blue
        (230, 230, 70),  # Yellow
    ]
    TARGET_COLORS = [
        (80, 30, 30),
        (30, 80, 30),
        (30, 50, 100),
        (90, 90, 30),
    ]
    COLOR_UI_TEXT = (220, 220, 220)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_ui = pygame.font.Font(None, 32)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # Etc...        
        
        # Initialize state variables
        self.moves_remaining = 0
        self.selector_pos = None
        self.blocks = []
        self.targets = []
        self.particles = []
        self.score = 0
        self.game_over = False
        self.win = False
        self.blocks_on_target_before_move = set()

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.moves_remaining = self.MAX_MOVES
        self.selector_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles = []
        
        self._generate_level()
        self.blocks_on_target_before_move = self._get_on_target_block_indices()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # Using a fixed, solvable level for consistency
        self.blocks = [
            {'pos': np.array([2, 2]), 'color_idx': 0, 'id': 0},
            {'pos': np.array([2, 5]), 'color_idx': 1, 'id': 1},
            {'pos': np.array([5, 3]), 'color_idx': 2, 'id': 2},
        ]
        self.targets = [
            {'pos': np.array([4, 2]), 'color_idx': 0},
            {'pos': np.array([4, 5]), 'color_idx': 1},
            {'pos': np.array([5, 5]), 'color_idx': 2},
        ]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Update game logic
        self.moves_remaining -= 1
        reward = 0
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        direction_vec = self._get_direction_vector(movement)

        dists_before = self._get_all_block_distances()
        self.blocks_on_target_before_move = self._get_on_target_block_indices()

        # --- Action Logic ---
        if space_held and movement != 0: # Push action
            block_idx_to_push = self._find_block_at(self.selector_pos)
            if block_idx_to_push is not None:
                # sfx: push_attempt
                self._push_block_chain(block_idx_to_push, direction_vec)
        else: # Selector move action
            new_pos = self.selector_pos + direction_vec
            if 0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE:
                self.selector_pos = new_pos
                # sfx: cursor_move
        
        # --- Reward Calculation ---
        reward += self._calculate_reward(dists_before)
        
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 100
                # sfx: level_complete
            else: # Ran out of moves
                reward -= 50
                # sfx: game_over
        
        self.score += reward
        # MUST return exactly this 5-tuple
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._draw_grid()
        self._draw_targets()
        self._draw_blocks()
        self._draw_selector()
        self._update_and_draw_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    # --- Helper Methods ---

    def _calculate_reward(self, dists_before):
        reward = 0
        
        # Distance-based reward
        dists_after = self._get_all_block_distances()
        for block_id in dists_before:
            if dists_after.get(block_id, 0) < dists_before.get(block_id, 0):
                reward += 1
            elif dists_after.get(block_id, 0) > dists_before.get(block_id, 0):
                reward -= 1
        
        # Event-based reward for placing a block on target
        blocks_on_target_after = self._get_on_target_block_indices()
        newly_on_target = blocks_on_target_after - self.blocks_on_target_before_move
        if newly_on_target:
            # sfx: block_on_target
            reward += 10 * len(newly_on_target)
            for block_id in newly_on_target:
                block_pos = next(b['pos'] for b in self.blocks if b['id'] == block_id)
                self._create_target_particles(block_pos)
        
        return reward

    def _get_direction_vector(self, movement):
        if movement == 1: return np.array([0, -1])  # Up
        if movement == 2: return np.array([0, 1])   # Down
        if movement == 3: return np.array([-1, 0])  # Left
        if movement == 4: return np.array([1, 0])   # Right
        return np.array([0, 0])

    def _find_block_at(self, pos):
        for i, block in enumerate(self.blocks):
            if np.array_equal(block['pos'], pos):
                return i
        return None

    def _push_block_chain(self, start_block_idx, direction):
        chain = []
        current_pos = self.blocks[start_block_idx]['pos'].copy()
        
        while True:
            idx = self._find_block_at(current_pos)
            if idx is not None:
                chain.append(idx)
                current_pos += direction
            else:
                break
        
        final_pos = self.blocks[chain[-1]]['pos'] + direction
        if not (0 <= final_pos[0] < self.GRID_SIZE and 0 <= final_pos[1] < self.GRID_SIZE):
            # sfx: push_fail_wall
            return False

        if self._find_block_at(final_pos) is not None:
             # sfx: push_fail_block
            return False

        for idx in reversed(chain):
            self.blocks[idx]['pos'] += direction
        # sfx: push_success
        return True

    def _get_all_block_distances(self):
        distances = {}
        for block in self.blocks:
            target_pos = next((t['pos'] for t in self.targets if t['color_idx'] == block['color_idx']), None)
            if target_pos is not None:
                dist = np.sum(np.abs(block['pos'] - target_pos))
                distances[block['id']] = dist
        return distances

    def _get_on_target_block_indices(self):
        on_target = set()
        for block in self.blocks:
            for target in self.targets:
                if block['color_idx'] == target['color_idx'] and np.array_equal(block['pos'], target['pos']):
                    on_target.add(block['id'])
                    break
        return on_target
        
    def _check_termination(self):
        if len(self._get_on_target_block_indices()) == len(self.blocks):
            self.win = True
            self.game_over = True
            return True
        if self.moves_remaining <= 0:
            self.game_over = True
            return True
        return False

    # --- Rendering Methods ---

    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_X_OFFSET + grid_pos[0] * self.CELL_SIZE
        y = self.GRID_Y_OFFSET + grid_pos[1] * self.CELL_SIZE
        return int(x), int(y)

    def _draw_grid(self):
        for i in range(self.GRID_SIZE + 1):
            start_x = self.GRID_X_OFFSET + i * self.CELL_SIZE
            start_y = self.GRID_Y_OFFSET
            end_x = start_x
            end_y = self.GRID_Y_OFFSET + self.GRID_SIZE * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (end_x, end_y), 1)

            start_x = self.GRID_X_OFFSET
            start_y = self.GRID_Y_OFFSET + i * self.CELL_SIZE
            end_x = self.GRID_X_OFFSET + self.GRID_SIZE * self.CELL_SIZE
            end_y = start_y
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (end_x, end_y), 1)

    def _draw_targets(self):
        for target in self.targets:
            px, py = self._grid_to_pixel(target['pos'])
            color = self.TARGET_COLORS[target['color_idx']]
            pygame.draw.rect(self.screen, color, (px, py, self.CELL_SIZE, self.CELL_SIZE))

    def _draw_blocks(self):
        for block in self.blocks:
            px, py = self._grid_to_pixel(block['pos'])
            color = self.BLOCK_COLORS[block['color_idx']]
            inset = 4
            pygame.draw.rect(self.screen, color, (px + inset, py + inset, self.CELL_SIZE - 2*inset, self.CELL_SIZE - 2*inset), border_radius=4)
            pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in color), (px + inset, py + inset, self.CELL_SIZE - 2*inset, self.CELL_SIZE - 2*inset), 2, border_radius=4)

    def _draw_selector(self):
        px, py = self._grid_to_pixel(self.selector_pos)
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
        thickness = int(2 + pulse * 2)
        
        rect = (px, py, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, rect, thickness, border_radius=6)

    def _render_ui(self):
        moves_text = f"Moves: {self.moves_remaining}"
        text_surface = self.font_ui.render(moves_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (15, 15))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = "PUZZLE SOLVED!" if self.win else "OUT OF MOVES"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            text_surface = self.font_game_over.render(end_text, True, color)
            text_rect = text_surface.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surface, text_rect)

    # --- Particle System ---

    def _create_target_particles(self, grid_pos):
        px, py = self._grid_to_pixel(grid_pos)
        center_x, center_y = px + self.CELL_SIZE // 2, py + self.CELL_SIZE // 2
        block = next((b for b in self.blocks if np.array_equal(b['pos'], grid_pos)), None)
        if block is None: return
        color = self.BLOCK_COLORS[block['color_idx']]
        
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append([[center_x, center_y], vel, self.PARTICLE_LIFESPAN, color])

    def _update_and_draw_particles(self):
        particles_to_keep = []
        for p in self.particles:
            p[0][0] += p[1][0] # pos.x += vel.x
            p[0][1] += p[1][1] # pos.y += vel.y
            p[2] -= 1 # life -= 1

            if p[2] > 0:
                alpha = int(255 * (p[2] / self.PARTICLE_LIFESPAN))
                pygame.gfxdraw.filled_circle(self.screen, int(p[0][0]), int(p[0][1]), 3, (*p[3], alpha))
                particles_to_keep.append(p)
        self.particles = particles_to_keep

    def _get_info(self):
        return {
            "score": self.score,
            "moves_remaining": self.moves_remaining,
            "win": self.win,
        }

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
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Block Pusher")
    clock = pygame.time.Clock()

    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Human Input Handling ---
        movement, space, shift = 0, 0, 0
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # We only step when a key is pressed because auto_advance is False
        should_step = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                continue
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE]:
                    should_step = True
                if event.key == pygame.K_r: # Reset on 'R'
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                    should_step = False
        
        if should_step:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS for human play

    print("Game Over!")
    # Keep the final screen visible for a moment
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                pygame.quit()
                exit()
        clock.tick(10)