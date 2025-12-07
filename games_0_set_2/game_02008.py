import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A block-pushing puzzle game where the player must move all colored blocks
    onto their corresponding targets within a limited number of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = "Controls: ↑↓←→ to push all blocks in the chosen direction."

    # User-facing game description
    game_description = "Push colored blocks onto their matching targets within the move limit."

    # Frames advance only on action
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    CELL_SIZE = 40
    NUM_BLOCKS = 3
    MAX_MOVES = 25

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_TEXT = (220, 230, 240)
    BLOCK_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
    ]
    TARGET_ALPHA = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 80)

        # --- Game State Variables ---
        self.blocks = []
        self.targets = []
        self.particles = []
        self.push_indicator = None
        self.steps = 0
        self.score = 0
        self.moves_remaining = 0
        self.game_over = False
        self.game_won = False

        # `reset()` is called here to initialize the state for the first time.
        # It's important that this happens after all attributes are initialized.
        # self.reset() # This was causing the error during initialization.
        # It's better practice to let the user call reset() before the first step.
        # However, to match the original code's behavior of being ready after __init__,
        # we will initialize the state but not call the full reset() which depends on a seeded RNG.
        self.np_random = np.random.default_rng() # Initialize a default RNG
        self._initialize_state()
        # self.validate_implementation() # This is for debugging and can be removed.

    def _initialize_state(self):
        """Initializes game state without a specific seed."""
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False
        self.game_won = False
        self.particles.clear()
        self.push_indicator = None

        self._generate_puzzle()
        self._update_block_target_status()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        reward = 0
        self.push_indicator = None

        if movement != 0:  # A push action was taken
            self.moves_remaining -= 1
            # Sound: # sfx_push.wav

            pre_move_distances = self._get_total_distance_to_targets()
            self._push_blocks(movement)
            post_move_distances = self._get_total_distance_to_targets()

            # Reward based on distance change (Manhattan distance)
            distance_change = pre_move_distances - post_move_distances
            if distance_change > 0:
                reward += 0.1 * distance_change
            elif distance_change < 0:
                reward += 0.2 * distance_change # Larger penalty for moving away

            # Check for newly completed targets
            newly_on_target = self._update_block_target_status()
            for block in newly_on_target:
                reward += 5
                # Sound: # sfx_block_on_target.wav
                self._create_particles(block['pos'], block['color'])

        # Check for termination conditions
        all_on_target = all(b['on_target'] for b in self.blocks)
        if all_on_target:
            self.game_won = True
            self.game_over = True
            reward += 50
            # Sound: # sfx_win_puzzle.wav
        elif self.moves_remaining <= 0:
            self.game_over = True
            reward -= 50
            # Sound: # sfx_lose_puzzle.wav

        self.score += reward
        self.steps += 1

        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,  # truncated is always False
            self._get_info(),
        )

    def _generate_puzzle(self):
        self.blocks.clear()
        self.targets.clear()
        occupied_coords = set()

        # Ensure puzzles are solvable by starting from solved state and moving blocks
        # For simplicity here, we place them randomly, which may be unsolvable.
        # A robust implementation would use a puzzle generator.
        for i in range(self.NUM_BLOCKS):
            color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]

            # Place block
            while True:
                pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
                if pos not in occupied_coords:
                    occupied_coords.add(pos)
                    break

            # Place target
            while True:
                target_pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
                if target_pos not in occupied_coords:
                    occupied_coords.add(target_pos)
                    break

            self.blocks.append({"pos": pos, "color": color, "target_pos": target_pos, "on_target": False})
            self.targets.append({"pos": target_pos, "color": color})

    def _push_blocks(self, direction):
        # 1:Up, 2:Down, 3:Left, 4:Right
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[direction]
        self.push_indicator = {'dir': (dx, dy), 'life': 5}

        # Sort blocks to handle chain reactions correctly
        sort_key = lambda b: b['pos'][0]
        reverse = (dx > 0)
        if dy != 0:
            sort_key = lambda b: b['pos'][1]
            reverse = (dy > 0)
        
        sorted_blocks = sorted(self.blocks, key=sort_key, reverse=reverse)
        block_pos_set = {b['pos'] for b in self.blocks}
        
        moved_any = False
        for block in sorted_blocks:
            current_pos = block['pos']
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)

            # Boundary check
            if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
                continue

            # Collision check with other blocks
            if next_pos in block_pos_set:
                continue

            # Move the block
            block['pos'] = next_pos
            block_pos_set.remove(current_pos)
            block_pos_set.add(next_pos)
            moved_any = True
        
        return moved_any

    def _update_block_target_status(self):
        newly_on_target = []
        for block in self.blocks:
            was_on_target = block.get('on_target', False)
            is_on_target = (block['pos'] == block['target_pos'])
            if is_on_target and not was_on_target:
                newly_on_target.append(block)
            block['on_target'] = is_on_target
        return newly_on_target

    def _get_total_distance_to_targets(self):
        total_dist = 0
        for block in self.blocks:
            dist = abs(block['pos'][0] - block['target_pos'][0]) + abs(block['pos'][1] - block['target_pos'][1])
            total_dist += dist
        return total_dist

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_targets()
        self._render_blocks()
        self._update_and_render_particles()
        self._render_push_indicator()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "blocks_on_target": sum(1 for b in self.blocks if b['on_target']),
        }

    def _render_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py))

    def _render_targets(self):
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 * 5
        for target in self.targets:
            rect = pygame.Rect(
                target['pos'][0] * self.CELL_SIZE,
                target['pos'][1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            target_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            fill_color = (*target['color'], self.TARGET_ALPHA // 4)
            pygame.draw.rect(target_surface, fill_color, (0, 0, self.CELL_SIZE, self.CELL_SIZE))
            outline_color = (*target['color'], int(self.TARGET_ALPHA + pulse * 20))
            pygame.draw.rect(target_surface, outline_color, (0, 0, self.CELL_SIZE, self.CELL_SIZE), width=int(3 + pulse / 2))
            self.screen.blit(target_surface, rect.topleft)

    def _render_blocks(self):
        for block in self.blocks:
            rect = pygame.Rect(
                block['pos'][0] * self.CELL_SIZE,
                block['pos'][1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            main_color = block['color']
            # FIX: Create tuples directly, not generators. Generators are exhausted
            # after one use, causing an error on the second draw call.
            shadow_color = tuple(max(0, c - 50) for c in main_color)
            highlight_color = tuple(min(255, c + 50) for c in main_color)
            
            inset = 4
            pygame.draw.rect(self.screen, shadow_color, rect)
            inner_rect = pygame.Rect(rect.left + inset, rect.top + inset, max(0, rect.width - inset * 2), max(0, rect.height - inset * 2))
            pygame.draw.rect(self.screen, main_color, inner_rect)
            
            # Now `highlight_color` is a tuple and can be used multiple times.
            pygame.draw.line(self.screen, highlight_color, (inner_rect.left, inner_rect.top), (inner_rect.right - 1, inner_rect.top), 2)
            pygame.draw.line(self.screen, highlight_color, (inner_rect.left, inner_rect.top), (inner_rect.left, inner_rect.bottom - 1), 2)
            
            if block['on_target']:
                center_x, center_y = rect.center
                points = [(center_x - 10, center_y), (center_x - 2, center_y + 8), (center_x + 10, center_y - 8)]
                pygame.draw.lines(self.screen, self.COLOR_TEXT, False, points, 4)

    def _create_particles(self, grid_pos, color):
        px, py = (grid_pos[0] + 0.5) * self.CELL_SIZE, (grid_pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({'pos': [px, py], 'vel': vel, 'color': color, 'life': lifespan, 'max_life': lifespan})

    def _update_and_render_particles(self):
        remaining_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95
            p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] > 0:
                remaining_particles.append(p)
                alpha = int(255 * (p['life'] / p['max_life']))
                color_with_alpha = (*p['color'], alpha)
                temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color_with_alpha, (2, 2), 2)
                self.screen.blit(temp_surf, (int(p['pos'][0] - 2), int(p['pos'][1] - 2)))
        self.particles = remaining_particles

    def _render_push_indicator(self):
        if self.push_indicator and self.push_indicator['life'] > 0:
            life_ratio = self.push_indicator['life'] / 5.0
            alpha = int(150 * life_ratio)
            dx, dy = self.push_indicator['dir']
            
            if dy == -1: rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 10)
            elif dy == 1: rect = pygame.Rect(0, self.SCREEN_HEIGHT - 10, self.SCREEN_WIDTH, 10)
            elif dx == -1: rect = pygame.Rect(0, 0, 10, self.SCREEN_HEIGHT)
            else: rect = pygame.Rect(self.SCREEN_WIDTH - 10, 0, 10, self.SCREEN_HEIGHT)

            indicator_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            indicator_surf.fill((255, 255, 255, alpha))
            self.screen.blit(indicator_surf, rect.topleft)

            self.push_indicator['life'] -= 1

    def _render_ui(self):
        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (15, 10))
        
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 10))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "PUZZLE SOLVED!" if self.game_won else "OUT OF MOVES"
            end_color = self.BLOCK_COLORS[1] if self.game_won else self.BLOCK_COLORS[0]
            end_text = self.font_large.render(end_text_str, True, end_color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")