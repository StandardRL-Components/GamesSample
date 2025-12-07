import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move. Push the colored boxes onto their matching targets before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, real-time puzzle game. Push all boxes onto their targets against the clock."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TIME_LIMIT = 30.0  # seconds
        self.MAX_STEPS = int(self.TIME_LIMIT * self.FPS)

        self.GRID_W, self.GRID_H = 16, 10
        self.CELL_SIZE = 40

        # --- Colors ---
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_GRID = (50, 50, 70)
        self.COLOR_WALL = (80, 80, 100)
        self.COLOR_PLAYER = (50, 150, 255)
        self.BOX_COLORS = [(255, 80, 80), (80, 255, 80), (255, 255, 80)]  # Red, Green, Yellow
        self.TARGET_COLORS = self.BOX_COLORS
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_WIN_TEXT = (100, 255, 100)
        self.COLOR_LOSE_TEXT = (255, 100, 100)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        # --- Game State (initialized in reset) ---
        self.player_pos = None
        self.box_positions = None
        self.target_positions = None
        self.boxes_on_target = None
        self.box_reward_given = None
        self.walls = None

        self.steps = 0
        self.score = 0
        self.time_left = 0.0
        self.game_over = False
        self.win_message = ""

        self.particles = []

        # --- Internal ---
        self._create_wall_layout()

        # Initialize the environment state so that validation can run.
        self.reset(seed=42)

        # --- Final Validation ---
        self.validate_implementation()

    def _create_wall_layout(self):
        self.walls = set()
        # Borders
        for x in range(self.GRID_W):
            self.walls.add((x, 0))
            self.walls.add((x, self.GRID_H - 1))
        for y in range(1, self.GRID_H - 1):
            self.walls.add((0, y))
            self.walls.add((self.GRID_W - 1, y))
        # Some internal walls for challenge
        for y in range(3, 7):
            self.walls.add((4, y))
            self.walls.add((self.GRID_W - 5, y))
        for x in range(5, self.GRID_W - 5):
            if x % 4 == 0:
                self.walls.add((x, 3))
                self.walls.add((x, self.GRID_H - 4))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.time_left = self.TIME_LIMIT
        self.game_over = False
        self.win_message = ""

        self.boxes_on_target = [False] * len(self.BOX_COLORS)
        self.box_reward_given = [False] * len(self.BOX_COLORS)

        # Generate puzzle layout
        valid_cells = []
        for x in range(1, self.GRID_W - 1):
            for y in range(1, self.GRID_H - 1):
                if (x, y) not in self.walls:
                    valid_cells.append((x, y))

        num_entities = 1 + len(self.BOX_COLORS) * 2
        if len(valid_cells) < num_entities:
            raise ValueError("Not enough valid cells to place all game entities.")

        chosen_indices = self.np_random.choice(len(valid_cells), size=num_entities, replace=False)
        chosen_cells = [valid_cells[i] for i in chosen_indices]

        self.player_pos = chosen_cells.pop(0)
        self.box_positions = [chosen_cells.pop(0) for _ in self.BOX_COLORS]
        self.target_positions = [chosen_cells.pop(0) for _ in self.TARGET_COLORS]

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = -0.01  # Small penalty for each step to encourage speed
        terminated = False

        if not self.game_over:
            self.time_left -= 1 / self.FPS

            # --- Handle Action ---
            movement = action[0]
            self._handle_movement(movement)

            # --- Update Game State & Rewards ---
            self._update_box_states()

            for i in range(len(self.BOX_COLORS)):
                if self.boxes_on_target[i] and not self.box_reward_given[i]:
                    reward += 1.0  # Reward for placing a box
                    self.box_reward_given[i] = True
                    # SFX: Box placed sound
                    self._create_particles(self.box_positions[i], self.BOX_COLORS[i])

            # --- Check Termination ---
            if all(self.boxes_on_target):
                self.game_over = True
                terminated = True
                reward += 10.0  # Win bonus
                self.win_message = "YOU WIN!"
                # SFX: Win fanfare
            elif self.time_left <= 0 or self.steps >= self.MAX_STEPS - 1:
                self.game_over = True
                terminated = True
                reward -= 10.0  # Lose penalty
                self.win_message = "TIME'S UP!"
                # SFX: Lose sound
        else:
            terminated = True
            reward = 0.0

        self.steps += 1
        self.score += reward

        self._update_particles()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        px, py = self.player_pos
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1  # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1  # Right

        if dx == 0 and dy == 0:
            return

        next_px, next_py = px + dx, py + dy

        if (next_px, next_py) in self.walls:
            # SFX: Bump wall
            return

        box_idx = self._get_box_at(next_px, next_py)
        if box_idx is not None:
            if self.boxes_on_target[box_idx]:
                return  # Cannot push a locked box

            behind_bx, behind_by = next_px + dx, next_py + dy

            if (behind_bx, behind_by) in self.walls or self._get_box_at(behind_bx, behind_by) is not None:
                # SFX: Bump box
                return

            self.box_positions[box_idx] = (behind_bx, behind_by)
            # SFX: Push box

        self.player_pos = (next_px, next_py)

    def _get_box_at(self, x, y):
        if self.box_positions is None: return None
        for i, pos in enumerate(self.box_positions):
            if pos == (x, y):
                return i
        return None

    def _update_box_states(self):
        for i in range(len(self.BOX_COLORS)):
            if self.box_positions[i] == self.target_positions[i]:
                self.boxes_on_target[i] = True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_W + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.HEIGHT))
        for y in range(self.GRID_H + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.WIDTH, py))

        # Draw walls
        if self.walls:
            for wx, wy in self.walls:
                rect = pygame.Rect(wx * self.CELL_SIZE, wy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_WALL, rect)

        # Draw targets
        if self.target_positions:
            for i, (tx, ty) in enumerate(self.target_positions):
                center_x = int(tx * self.CELL_SIZE + self.CELL_SIZE / 2)
                center_y = int(ty * self.CELL_SIZE + self.CELL_SIZE / 2)
                radius = int(self.CELL_SIZE * 0.35)
                color = self.TARGET_COLORS[i]
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)

        # Draw boxes
        if self.box_positions:
            for i, (bx, by) in enumerate(self.box_positions):
                rect = pygame.Rect(bx * self.CELL_SIZE + 2, by * self.CELL_SIZE + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
                color = self.BOX_COLORS[i]
                pygame.draw.rect(self.screen, color, rect, border_radius=4)
                if self.boxes_on_target[i]:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect, width=3, border_radius=4)

        # Draw player
        if self.player_pos:
            px, py = self.player_pos
            rect = pygame.Rect(px * self.CELL_SIZE + 4, py * self.CELL_SIZE + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=8)
            pygame.draw.rect(self.screen, (200, 220, 255), rect.inflate(4, 4), width=2, border_radius=10)

        # Draw particles
        for p in self.particles:
            pos = [int(p['pos'][0]), int(p['pos'][1])]
            size = int(p['life'] / p['max_life'] * 5)
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], pos, size)

    def _render_ui(self):
        score_surf = self.font_small.render(f"Score: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        timer_width = 200
        timer_height = 20
        timer_x = self.WIDTH - timer_width - 10
        timer_y = 10
        time_ratio = max(0, self.time_left / self.TIME_LIMIT)

        pygame.draw.rect(self.screen, self.COLOR_GRID, (timer_x, timer_y, timer_width, timer_height), border_radius=4)
        fill_color = (255, 255, 0) if time_ratio > 0.5 else (255, 165, 0) if time_ratio > 0.2 else (255, 0, 0)
        if time_ratio > 0:
            pygame.draw.rect(self.screen, fill_color, (timer_x, timer_y, int(timer_width * time_ratio), timer_height),
                             border_radius=4)

        if self.game_over:
            color = self.COLOR_WIN_TEXT if "WIN" in self.win_message else self.COLOR_LOSE_TEXT
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg_surf = self.font_large.render(self.win_message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "boxes_on_target": sum(self.boxes_on_target)
        }

    def _create_particles(self, grid_pos, color):
        # SFX: Particle burst
        center_x = (grid_pos[0] + 0.5) * self.CELL_SIZE
        center_y = (grid_pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            life = random.randint(15, 30)  # in frames
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life, 'max_life': life, 'color': color
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.98  # friction
            p['vel'][1] *= 0.98  # friction
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3), f"Obs shape is {test_obs.shape}, expected (400, 640, 3)"
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")