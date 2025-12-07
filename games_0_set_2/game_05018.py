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

    user_guide = (
        "Controls: ↑↓←→ to move the cursor and rake the sand. "
        "Press Space to plant a pink flower, or Shift to plant a purple flower."
    )

    game_description = (
        "Create a tranquil Zen garden by raking sand and strategically planting flowers "
        "to maximize your aesthetic score before time runs out."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # --- Game Constants ---
        self.BORDER = 40
        self.GARDEN_RECT = pygame.Rect(
            self.BORDER, self.BORDER,
            self.WIDTH - 2 * self.BORDER, self.HEIGHT - 2 * self.BORDER
        )
        self.CELL_SIZE = 16
        self.GRID_W = self.GARDEN_RECT.width // self.CELL_SIZE
        self.GRID_H = self.GARDEN_RECT.height // self.CELL_SIZE

        self.MAX_TIME = 300.0  # 5 minutes
        self.MAX_STEPS = 9000  # 300s * 30fps
        self.MAX_SCORE = 100

        # --- Colors ---
        self.COLOR_BG = (20, 20, 20)
        self.COLOR_WALL = (60, 60, 65)
        self.COLOR_SAND = (210, 195, 160)
        self.COLOR_RAKE_LINE = (180, 165, 130)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_CURSOR_PLANT1 = (255, 150, 200)
        self.COLOR_CURSOR_PLANT2 = (200, 150, 255)
        self.COLOR_FLOWER1 = (255, 105, 180)  # Pink
        self.COLOR_FLOWER1_CENTER = (255, 255, 0)
        self.COLOR_FLOWER2 = (147, 112, 219)  # Purple
        self.COLOR_FLOWER2_CENTER = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (30, 30, 30)

        # --- Fonts ---
        try:
            self.font_ui = pygame.font.Font("freesansbold.ttf", 20)
            self.font_msg = pygame.font.Font("freesansbold.ttf", 48)
        except FileNotFoundError:
            self.font_ui = pygame.font.Font(None, 24)
            self.font_msg = pygame.font.Font(None, 50)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_reason = ""
        self.timer = 0.0
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_speed = 3
        self.sand_grid = np.zeros((1, 1), dtype=np.uint8)
        self.flowers = []
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False

        # self.reset() is called implicitly by super().__init__()
        # and then again explicitly if needed.
        # We initialize here to ensure attributes exist before reset is called.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_reason = ""
        self.timer = self.MAX_TIME

        self.player_pos = pygame.math.Vector2(self.GARDEN_RECT.center)

        self.sand_grid = np.zeros((self.GRID_W, self.GRID_H), dtype=np.uint8)
        self.flowers = []
        self.particles = []

        self.prev_space_held = True  # Prevent planting on first frame
        self.prev_shift_held = True

        return self._get_observation(), self._get_info()

    def _grid_coords(self, pos):
        x = int((pos.x - self.GARDEN_RECT.left) / self.CELL_SIZE)
        y = int((pos.y - self.GARDEN_RECT.top) / self.CELL_SIZE)
        return np.clip(x, 0, self.GRID_W - 1), np.clip(y, 0, self.GRID_H - 1)

    def _pixel_coords(self, grid_x, grid_y):
        x = self.GARDEN_RECT.left + (grid_x + 0.5) * self.CELL_SIZE
        y = self.GARDEN_RECT.top + (grid_y + 0.5) * self.CELL_SIZE
        return pygame.math.Vector2(x, y)

    def step(self, action):
        reward = 0
        self.steps += 1

        if not self.game_over:
            self.timer = max(0, self.timer - 1.0 / 30.0)

            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

            reward += self._handle_movement_and_raking(movement)
            reward += self._handle_planting(space_held, shift_held)

            self.prev_space_held = space_held
            self.prev_shift_held = shift_held

            self._update_game_elements()

            new_score = self._calculate_aesthetics()
            # Reward for score increase, but don't penalize decreases
            reward += max(0, (new_score - self.score) * 0.5)
            self.score = new_score

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and self.termination_reason == "Score Reached":
            reward = self.MAX_SCORE
        elif terminated and self.termination_reason == "Time's Up":
            reward = -self.MAX_SCORE

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_movement_and_raking(self, movement):
        old_pos = pygame.math.Vector2(self.player_pos)
        old_grid_pos = self._grid_coords(old_pos)

        if movement == 1: self.player_pos.y -= self.player_speed
        elif movement == 2: self.player_pos.y += self.player_speed
        elif movement == 3: self.player_pos.x -= self.player_speed
        elif movement == 4: self.player_pos.x += self.player_speed

        self.player_pos.x = np.clip(self.player_pos.x, self.GARDEN_RECT.left, self.GARDEN_RECT.right)
        self.player_pos.y = np.clip(self.player_pos.y, self.GARDEN_RECT.top, self.GARDEN_RECT.bottom)

        new_grid_pos = self._grid_coords(self.player_pos)

        if old_grid_pos != new_grid_pos and movement != 0:
            dx = new_grid_pos[0] - old_grid_pos[0]
            dy = new_grid_pos[1] - old_grid_pos[1]
            rake_dir = 0
            if abs(dx) > 0: rake_dir = 2  # Horizontal
            if abs(dy) > 0: rake_dir = 1  # Vertical

            gx, gy = old_grid_pos
            if 0 <= gx < self.GRID_W and 0 <= gy < self.GRID_H:
                if self.sand_grid[gx, gy] == 0:
                    self.sand_grid[gx, gy] = rake_dir
                    self._spawn_particles(self._pixel_coords(gx, gy), 5)
                    return 0.1  # Reward for new line
                elif self.sand_grid[gx, gy] != rake_dir and self.sand_grid[gx, gy] != 3:
                    self.sand_grid[gx, gy] = 3  # Create intersection
                    self._spawn_particles(self._pixel_coords(gx, gy), 8)
                    return 0.1
        return 0

    def _handle_planting(self, space_held, shift_held):
        reward = 0
        plant_type = 0
        if space_held and not self.prev_space_held:
            plant_type = 1
        elif shift_held and not self.prev_shift_held:
            plant_type = 2

        if plant_type > 0:
            can_plant = True
            for flower in self.flowers:
                if flower['pos'].distance_to(self.player_pos) < self.CELL_SIZE * 1.5:
                    can_plant = False
                    break
            if can_plant:
                # sfx: plant_flower
                old_score = self._calculate_aesthetics()
                self.flowers.append({
                    'pos': pygame.math.Vector2(self.player_pos),
                    'type': plant_type,
                    'growth': 0.1,  # Start small
                    'id': self.np_random.random()
                })
                new_score = self._calculate_aesthetics()
                if new_score > old_score:
                    reward += 1.0
        return reward

    def _update_game_elements(self):
        # Update flower growth
        for f in self.flowers:
            f['growth'] = min(1.0, f['growth'] + 0.05)

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _calculate_aesthetics(self):
        if not self.flowers:
            return 0

        score = 0.0
        # Rake bonus: +0.1 for each raked tile
        score += np.count_nonzero(self.sand_grid) * 0.05

        flower_positions = [f['pos'] for f in self.flowers]

        for i, f1 in enumerate(self.flowers):
            # Bonus for placement on special tiles
            gx, gy = self._grid_coords(f1['pos'])
            if self.sand_grid[gx, gy] == 3: score += 3.0  # Intersection
            elif self.sand_grid[gx, gy] > 0: score += 1.5  # Any raked line

            # Penalize proximity and reward symmetry
            for j in range(i + 1, len(self.flowers)):
                f2 = self.flowers[j]
                dist = f1['pos'].distance_to(f2['pos'])
                # Proximity penalty
                score -= 60 / max(dist, 10)

                # Horizontal symmetry bonus
                mirrored_x = self.GARDEN_RECT.centerx + (self.GARDEN_RECT.centerx - f2['pos'].x)
                if abs(f1['pos'].x - mirrored_x) < self.CELL_SIZE and abs(f1['pos'].y - f2['pos'].y) < self.CELL_SIZE:
                    score += 8.0

        # Variety Bonus
        if self.flowers:
            num_type1 = sum(1 for f in self.flowers if f['type'] == 1)
            num_type2 = len(self.flowers) - num_type1
            balance = 1.0 - abs(num_type1 - num_type2) / len(self.flowers)
            score += balance * len(self.flowers) * 1.5

        # Scale score non-linearly to make 100 achievable
        # Using a logistic-like function to map raw score to 0-100 range
        # Tuned to make a few well-placed flowers score well.
        if score > -300: # Avoid math domain error with large negative scores
            scaled_score = self.MAX_SCORE / (1 + math.exp(-0.1 * (score - 30)))
        else:
            scaled_score = 0
        return int(np.clip(scaled_score, 0, self.MAX_SCORE))

    def _check_termination(self):
        if self.game_over:
            return True
        if self.score >= self.MAX_SCORE:
            self.game_over = True
            self.termination_reason = "Score Reached"
            return True
        if self.timer <= 0:
            self.game_over = True
            self.termination_reason = "Time's Up"
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.HEIGHT))
        # Draw Sand
        pygame.draw.rect(self.screen, self.COLOR_SAND, self.GARDEN_RECT)

        # Draw Rake Lines
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                state = self.sand_grid[x, y]
                if state > 0:
                    px, py = self._pixel_coords(x, y)
                    if state == 1 or state == 3:  # Vertical
                        start = (int(px), int(py - self.CELL_SIZE / 2))
                        end = (int(px), int(py + self.CELL_SIZE / 2))
                        pygame.draw.aaline(self.screen, self.COLOR_RAKE_LINE, start, end, 2)
                    if state == 2 or state == 3:  # Horizontal
                        start = (int(px - self.CELL_SIZE / 2), int(py))
                        end = (int(px + self.CELL_SIZE / 2), int(py))
                        pygame.draw.aaline(self.screen, self.COLOR_RAKE_LINE, start, end, 2)

        # Draw Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((p['size'] * 2, p['size'] * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))

        # Draw Flowers
        for f in self.flowers:
            pos = (int(f['pos'].x), int(f['pos'].y))
            size = int(self.CELL_SIZE * 0.6 * f['growth'])
            center_size = int(size * 0.4)
            color = self.COLOR_FLOWER1 if f['type'] == 1 else self.COLOR_FLOWER2
            center_color = self.COLOR_FLOWER1_CENTER if f['type'] == 1 else self.COLOR_FLOWER2_CENTER

            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], center_size, center_color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], center_size, center_color)

        # Draw Player Cursor
        if not self.game_over:
            cursor_color = self.COLOR_CURSOR
            if self.prev_space_held: cursor_color = self.COLOR_CURSOR_PLANT1
            elif self.prev_shift_held: cursor_color = self.COLOR_CURSOR_PLANT2

            pos = (int(self.player_pos.x), int(self.player_pos.y))
            radius = int(self.CELL_SIZE * 0.5)
            # Glow effect
            temp_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*cursor_color, 50), (radius * 2, radius * 2), radius * 2)
            pygame.draw.circle(temp_surf, (*cursor_color, 100), (radius * 2, radius * 2), int(radius * 1.5))
            self.screen.blit(temp_surf, (pos[0] - radius * 2, pos[1] - radius * 2))

            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, cursor_color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius - 1, cursor_color)

    def _render_text(self, text, font, pos, color=None, shadow_color=None):
        color = color or self.COLOR_TEXT
        shadow_color = shadow_color or self.COLOR_TEXT_SHADOW
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect(center=pos)
        shadow_rect = shadow_surf.get_rect(center=(pos[0] + 2, pos[1] + 2))
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score
        score_text = f"Aesthetic: {self.score} / {self.MAX_SCORE}"
        self._render_text(score_text, self.font_ui, (120, 20))

        # Timer
        mins, secs = divmod(int(self.timer), 60)
        timer_text = f"Time: {mins:02d}:{secs:02d}"
        self._render_text(timer_text, self.font_ui, (self.WIDTH - 90, 20))

        # Game Over Message
        if self.game_over:
            msg = ""
            if self.termination_reason == "Score Reached":
                msg = "Garden Complete!"
            elif self.termination_reason == "Time's Up":
                msg = "Time's Up"

            if msg:
                self._render_text(msg, self.font_msg, self.screen.get_rect().center)

    def _spawn_particles(self, pos, count):
        # sfx: rake_sand
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': vel,
                'life': life,
                'max_life': life,
                'size': self.np_random.integers(1, 4),
                'color': self.COLOR_RAKE_LINE
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "termination_reason": self.termination_reason,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It is not used for the evaluation, but is helpful for testing.
    # Re-enable the display driver for interactive mode.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv()
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Zen Garden")

    terminated = False
    truncated = False
    clock = pygame.time.Clock()

    print(env.game_description)
    print(env.user_guide)

    while not terminated and not truncated:
        movement = 0  # no-op
        space_held = 0
        shift_held = 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        # Keyboard input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30)  # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Reason: {info.get('termination_reason', 'N/A')}")
    env.close()