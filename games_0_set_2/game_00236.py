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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use Space/Shift to cycle through blocks. Use arrow keys to push the selected block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down puzzle game. Push colored blocks into their matching goal zones before the time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 40
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.GRID_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.GRID_SIZE
        self.NUM_BLOCKS = 8
        self.MAX_STEPS = 900

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_WALL = (10, 20, 30)
        self.BLOCK_COLORS = [
            (255, 87, 87),    # Red
            (87, 255, 87),    # Green
            (87, 87, 255),    # Blue
            (255, 255, 87),   # Yellow
            (255, 87, 255),   # Magenta
            (87, 255, 255),   # Cyan
            (255, 165, 0),    # Orange
            (128, 0, 128),    # Purple
        ]
        self.GOAL_COLORS = [pygame.Color(c).lerp(self.COLOR_BG, 0.6) for c in self.BLOCK_COLORS]
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_SELECTOR = (255, 255, 255)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 24)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.blocks = []
        self.goals = []
        self.selected_block_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.rng = None

        # The initial reset will seed the RNG
        self.reset()

    def _generate_initial_state(self):
        """Generates a random, non-overlapping initial state for blocks and goals."""
        possible_coords = [
            (x, y) for x in range(1, self.GRID_WIDTH - 1) for y in range(1, self.GRID_HEIGHT - 1)
        ]
        
        chosen_indices = self.rng.choice(
            len(possible_coords),
            size=self.NUM_BLOCKS * 2,
            replace=False
        )
        
        chosen_coords = [possible_coords[i] for i in chosen_indices]

        goal_coords = chosen_coords[:self.NUM_BLOCKS]
        block_coords = chosen_coords[self.NUM_BLOCKS:]

        self.goals = []
        self.blocks = []
        for i in range(self.NUM_BLOCKS):
            self.goals.append({
                "pos": pygame.Vector2(goal_coords[i]),
                "color": self.GOAL_COLORS[i],
                "id": i
            })
            self.blocks.append({
                "pos": pygame.Vector2(block_coords[i]),
                "color": self.BLOCK_COLORS[i],
                "id": i,
                "is_in_goal": False
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Use the seeded random number generator from Gymnasium
        self.rng = self.np_random

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_block_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        
        self._generate_initial_state()
        self._update_goal_status()

        return self._get_observation(), self._get_info()

    def _handle_push(self, block_idx, direction, pushed_indices):
        """Recursively attempts to push a block and any subsequent blocks."""
        if block_idx in pushed_indices:
            return False

        pushed_indices.add(block_idx)
        
        block = self.blocks[block_idx]
        target_pos = block['pos'] + direction

        if not (0 <= target_pos.x < self.GRID_WIDTH and 0 <= target_pos.y < self.GRID_HEIGHT):
            return False

        for i, other_block in enumerate(self.blocks):
            if i == block_idx:
                continue
            if other_block['pos'] == target_pos:
                if not self._handle_push(i, direction, pushed_indices):
                    return False

        block['pos'] = target_pos
        return True

    def _update_goal_status(self):
        """Checks which blocks are in their correct goals and returns the number of newly scored blocks."""
        newly_scored = 0
        for block in self.blocks:
            is_now_in_goal = False
            for goal in self.goals:
                if block['id'] == goal['id'] and block['pos'] == goal['pos']:
                    is_now_in_goal = True
                    break
            
            if is_now_in_goal and not block['is_in_goal']:
                newly_scored += 1
                # sound: goal_achieved.wav
            
            block['is_in_goal'] = is_now_in_goal
        return newly_scored

    def _create_particles(self, pos, color, count):
        """Create a burst of particles."""
        grid_center_x = pos.x * self.GRID_SIZE + self.GRID_SIZE / 2
        grid_center_y = pos.y * self.GRID_SIZE + self.GRID_SIZE / 2
        for _ in range(count):
            self.particles.append({
                "pos": pygame.Vector2(grid_center_x, grid_center_y),
                "vel": pygame.Vector2(self.rng.uniform(-2, 2), self.rng.uniform(-2, 2)),
                "radius": self.rng.uniform(2, 5),
                "color": color,
                "lifetime": self.rng.integers(15, 30)
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        
        if space_held and not self.last_space_held:
            self.selected_block_idx = (self.selected_block_idx + 1) % self.NUM_BLOCKS
            # sound: select.wav
        if shift_held and not self.last_shift_held:
            self.selected_block_idx = (self.selected_block_idx - 1 + self.NUM_BLOCKS) % self.NUM_BLOCKS
            # sound: select.wav

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        move_made = False
        if movement > 0:
            self.steps += 1
            reward -= 0.1

            direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            direction = pygame.Vector2(direction_map[movement])
            
            original_pos = self.blocks[self.selected_block_idx]['pos'].copy()
            
            if self._handle_push(self.selected_block_idx, direction, set()):
                move_made = True
                # sound: push.wav
                self._create_particles(original_pos, self.blocks[self.selected_block_idx]['color'], 20)

        if move_made:
            newly_scored_count = self._update_goal_status()
            reward += newly_scored_count * 1.0
            self.score += newly_scored_count * 1.0
        
        if all(b['is_in_goal'] for b in self.blocks):
            terminated = True
            self.game_over = True
            reward += 100
            self.score += 100
            # sound: win.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            reward -= 50
            self.score -= 50
            # sound: lose.wav

        truncated = False
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_WALL, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_WALL, (0, y), (self.SCREEN_WIDTH, y))

        for goal in self.goals:
            rect = pygame.Rect(
                goal['pos'].x * self.GRID_SIZE,
                goal['pos'].y * self.GRID_SIZE,
                self.GRID_SIZE, self.GRID_SIZE
            )
            pygame.draw.rect(self.screen, goal['color'], rect)
            pygame.draw.rect(self.screen, tuple(c*0.8 for c in goal['color']), rect, 2)

        for i, block in enumerate(self.blocks):
            rect = pygame.Rect(
                block['pos'].x * self.GRID_SIZE,
                block['pos'].y * self.GRID_SIZE,
                self.GRID_SIZE, self.GRID_SIZE
            )
            shadow_color = tuple(c*0.5 for c in block['color'])
            highlight_color = tuple(min(255, c*1.2) for c in block['color'])
            pygame.draw.rect(self.screen, shadow_color, rect.move(2, 2))
            pygame.draw.rect(self.screen, block['color'], rect)
            pygame.draw.rect(self.screen, highlight_color, rect.inflate(-self.GRID_SIZE*0.7, -self.GRID_SIZE*0.7))

            if block['is_in_goal']:
                center = (int(rect.centerx), int(rect.centery))
                pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
                radius = int(self.GRID_SIZE * 0.15 * pulse + self.GRID_SIZE * 0.05)
                pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, (255, 255, 255, 180))
                pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, (255, 255, 255, 180))

        if not self.game_over:
            selected_block = self.blocks[self.selected_block_idx]
            rect = pygame.Rect(
                selected_block['pos'].x * self.GRID_SIZE,
                selected_block['pos'].y * self.GRID_SIZE,
                self.GRID_SIZE, self.GRID_SIZE
            )
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
            line_width = int(pulse * 2 + 2)
            pygame.draw.rect(self.screen, self.COLOR_SELECTOR, rect, line_width)

        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] -= 0.1
            if p['lifetime'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['lifetime'] / 30))
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((int(p['radius']*2), int(p['radius']*2)), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
                self.screen.blit(temp_surf, p['pos'] - pygame.Vector2(p['radius'], p['radius']))

    def _render_ui(self):
        time_left = max(0, self.MAX_STEPS - self.steps)
        timer_text = self.font_large.render(f"{time_left}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 10))
        self.screen.blit(timer_text, timer_rect)

        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topleft=(15, 10))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            win_condition = all(b['is_in_goal'] for b in self.blocks)
            msg = "COMPLETE!" if win_condition else "TIME UP!"
            
            end_text = self.font_large.render(msg, True, self.COLOR_SELECTOR)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks_in_goal": sum(1 for b in self.blocks if b['is_in_goal']),
            "max_steps": self.MAX_STEPS,
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()