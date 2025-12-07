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
        "Controls: Arrow keys to move between hiding spots. Space to perform a quick dash."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A side-scrolling stealth horror game. Sneak past patrolling guards to reach the exit. Don't get caught in their vision."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.GRID_SIZE = 40
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE

        # --- Colors (Dark, limited palette) ---
        self.COLOR_BG = (10, 8, 12)
        self.COLOR_WALL = (40, 35, 45)
        self.COLOR_HIDING_SPOT = (60, 55, 70)
        self.COLOR_PLAYER = (230, 230, 255)
        self.COLOR_PLAYER_GLOW = (150, 150, 255)
        self.COLOR_ENEMY = (200, 40, 40)
        self.COLOR_VISION_CONE = (180, 50, 50)
        self.COLOR_EXIT = (40, 200, 120)
        self.COLOR_UI_TEXT = (200, 200, 220)
        self.COLOR_GRAIN = (25, 25, 35)

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
        try:
            self.font_small = pygame.font.SysFont("monospace", 16)
            self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 20)
            self.font_large = pygame.font.Font(None, 60)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.hiding_spots = []
        self.enemies = []
        self.level_grid = np.array([])
        self.light_flicker = 1.0
        self.last_move_dir = (0, 0)
        self.np_random = None

        # self.reset() # This is called by the wrapper/runner
        # self.validate_implementation() # This is for testing and not part of a standard __init__

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.last_move_dir = (1, 0)  # Default to right

        self._generate_level()

        # Find a starting spot for the player
        start_candidates = [spot for spot in self.hiding_spots if spot[0] < self.GRID_W // 4]
        if start_candidates:
            self.player_pos = start_candidates[self.np_random.integers(len(start_candidates))]
        else:
            self.player_pos = self.hiding_spots[0]

        self._initialize_enemies()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]
        space_pressed = action[1] == 1
        # shift_pressed = action[2] == 1 # Unused

        reward = 0
        terminated = False

        prev_dist_to_exit = self._distance(self.player_pos, self.exit_pos)

        # --- Handle Player Action ---
        if space_pressed:
            # Sneak/Dash action
            # SFX: Player dash whoosh
            dash_target_grid = (self.player_pos[0] + self.last_move_dir[0] * 2, self.player_pos[1] + self.last_move_dir[1] * 2)
            if self._is_valid_and_empty(dash_target_grid):
                self.player_pos = dash_target_grid
            # Dash has a higher risk of detection
        elif movement != 0:
            # Move to adjacent hiding spot
            move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            move_dir = move_map.get(movement, (0, 0))
            if move_dir != (0, 0):
                self.last_move_dir = move_dir
                target_pos = (self.player_pos[0] + move_dir[0], self.player_pos[1] + move_dir[1])
                if target_pos in self.hiding_spots:
                    self.player_pos = target_pos
                    # SFX: Player footstep
                else:
                    # SFX: Bump into wall
                    pass  # Invalid move, do nothing

        # --- Update Game State ---
        self._update_enemies()

        # --- Calculate Rewards & Check Termination ---
        new_dist_to_exit = self._distance(self.player_pos, self.exit_pos)
        if new_dist_to_exit < prev_dist_to_exit:
            reward += 0.1  # Closer to exit

        in_danger = False
        for enemy in self.enemies:
            if self._is_in_vision_cone(self.player_pos, enemy):
                in_danger = True
                if enemy['alert_level'] > 0.8:  # Detected if alert is high
                    reward = -10
                    terminated = True
                    self.game_over = True
                    self.win_message = "DETECTED"
                    # SFX: Loud detection alarm
                    break

        if not terminated and in_danger:
            reward -= 0.2  # Penalty for being in a vision cone

        if not terminated and self.player_pos == self.exit_pos:
            reward = 100
            terminated = True
            self.game_over = True
            self.win_message = "ESCAPED"
            # SFX: Success chime

        self.steps += 1
        if not terminated and self.steps >= self.MAX_STEPS:
            reward = -5  # Penalty for running out of time
            terminated = True
            self.game_over = True
            self.win_message = "OUT OF TIME"

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _get_observation(self):
        # --- Update visual effects ---
        if self.np_random:
            self.light_flicker = 0.8 + self.np_random.random() * 0.2

        # --- Rendering ---
        self.screen.fill(self._modulate_color(self.COLOR_BG))
        self._render_game()
        self._render_ui()

        # --- Film Grain ---
        if self.np_random:
            for _ in range(200):
                x = self.np_random.integers(0, self.WIDTH)
                y = self.np_random.integers(0, self.HEIGHT)
                pygame.gfxdraw.pixel(self.screen, x, y, self.COLOR_GRAIN)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
            "distance_to_exit": self._distance(self.player_pos, self.exit_pos)
        }

    def _modulate_color(self, color, factor=1.0):
        """Modulates color brightness based on global light flicker."""
        r, g, b = color
        f = self.light_flicker * factor
        return (max(0, min(255, int(r * f))),
                max(0, min(255, int(g * f))),
                max(0, min(255, int(b * f))))

    def _render_game(self):
        # Render walls, hiding spots, and exit
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                rect = pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                cell_type = self.level_grid[y, x]
                if cell_type == 1:  # Wall
                    pygame.draw.rect(self.screen, self._modulate_color(self.COLOR_WALL), rect)
                elif cell_type == 2:  # Hiding spot
                    pygame.draw.circle(self.screen, self._modulate_color(self.COLOR_HIDING_SPOT), rect.center,
                                       self.GRID_SIZE // 4)
                elif cell_type == 3:  # Exit
                    pygame.draw.rect(self.screen, self._modulate_color(self.COLOR_EXIT), rect.inflate(-10, -10))

        # Render enemies and vision cones
        for enemy in self.enemies:
            self._render_enemy(enemy)

        # Render player
        px, py = self.player_pos
        player_center = (int((px + 0.5) * self.GRID_SIZE), int((py + 0.5) * self.GRID_SIZE))

        # Glow effect
        glow_radius = int(self.GRID_SIZE * 0.5)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (player_center[0] - glow_radius, player_center[1] - glow_radius))

        pygame.draw.circle(self.screen, self.COLOR_PLAYER, player_center, self.GRID_SIZE // 3)

    def _render_enemy(self, enemy):
        ex, ey = enemy['pos']
        enemy_center = (int((ex + 0.5) * self.GRID_SIZE), int((ey + 0.5) * self.GRID_SIZE))

        # Vision Cone
        vision_angle = 45  # degrees
        vision_range = enemy['vision_range'] * self.GRID_SIZE
        angle_rad = math.atan2(enemy['dir'][1], enemy['dir'][0])

        p1 = enemy_center
        p2 = (enemy_center[0] + vision_range * math.cos(angle_rad - math.radians(vision_angle / 2)),
              enemy_center[1] + vision_range * math.sin(angle_rad - math.radians(vision_angle / 2)))
        p3 = (enemy_center[0] + vision_range * math.cos(angle_rad + math.radians(vision_angle / 2)),
              enemy_center[1] + vision_range * math.sin(angle_rad + math.radians(vision_angle / 2)))

        cone_color = self._modulate_color(self.COLOR_VISION_CONE)
        alpha_color = (*cone_color, int(50 + 100 * enemy['alert_level']))
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], alpha_color)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], alpha_color)

        # Enemy Body
        enemy_color = self._modulate_color(self.COLOR_ENEMY)
        size = self.GRID_SIZE // 3
        points = [
            (enemy_center[0] + size * math.cos(angle_rad), enemy_center[1] + size * math.sin(angle_rad)),
            (enemy_center[0] + size * math.cos(angle_rad + 2.5), enemy_center[1] + size * math.sin(angle_rad + 2.5)),
            (enemy_center[0] + size * math.cos(angle_rad - 2.5), enemy_center[1] + size * math.sin(angle_rad - 2.5)),
        ]
        pygame.draw.polygon(self.screen, enemy_color, points)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        dist = self._distance(self.player_pos, self.exit_pos)
        dist_text = self.font_small.render(f"DIST TO EXIT: {dist:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(dist_text, (10, 30))

        if self.game_over:
            # Red flash on detection
            if self.win_message == "DETECTED":
                s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                s.fill((255, 0, 0, 50))
                self.screen.blit(s, (0, 0))

            end_text = self.font_large.render(self.win_message, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _generate_level(self):
        self.level_grid = np.zeros((self.GRID_H, self.GRID_W), dtype=int)
        self.hiding_spots = []

        # Create borders
        self.level_grid[0, :] = 1
        self.level_grid[-1, :] = 1
        self.level_grid[:, 0] = 1
        self.level_grid[:, -1] = 1

        # Generate some random walls
        for _ in range(int(self.GRID_W * self.GRID_H * 0.1)):
            x, y = self.np_random.integers(1, [self.GRID_W - 1, self.GRID_H - 1])
            self.level_grid[y, x] = 1

        # Generate hiding spots in empty spaces
        for y in range(1, self.GRID_H - 1):
            for x in range(1, self.GRID_W - 1):
                if self.level_grid[y, x] == 0:
                    # More likely to be a hiding spot if neighbors are empty
                    if self.np_random.random() < 0.4:
                        self.level_grid[y, x] = 2
                        self.hiding_spots.append((x, y))

        if not self.hiding_spots:  # Failsafe
            self.hiding_spots.append((2, self.GRID_H // 2))
            self.level_grid[self.GRID_H // 2, 2] = 2

        # Place Exit
        exit_candidates = [spot for spot in self.hiding_spots if spot[0] > self.GRID_W * 0.75]
        if exit_candidates:
            self.exit_pos = exit_candidates[self.np_random.integers(len(exit_candidates))]
        else:  # Failsafe exit
            self.exit_pos = self.hiding_spots[-1]
        self.level_grid[self.exit_pos[1], self.exit_pos[0]] = 3

    def _initialize_enemies(self):
        self.enemies = []
        num_enemies = self.np_random.integers(2, 5)
        if not self.hiding_spots:
            return
            
        for _ in range(num_enemies):
            spots_arr = np.array(self.hiding_spots)
            path = self.np_random.choice(spots_arr, axis=0, size=2, replace=False).tolist()
            enemy = {
                'pos': list(path[0]),
                'path': path,
                'path_index': 0,
                'dir': (0, 0),
                'speed': self.np_random.uniform(0.05, 0.1),
                'vision_range': self.np_random.uniform(3, 5),
                'alert_level': 0.0
            }
            self.enemies.append(enemy)

    def _update_enemies(self):
        base_speed_increase = max(0, self.steps - 50) * 0.0002
        for enemy in self.enemies:
            target_pos = enemy['path'][enemy['path_index']]

            dx = target_pos[0] - enemy['pos'][0]
            dy = target_pos[1] - enemy['pos'][1]
            dist = math.sqrt(dx ** 2 + dy ** 2)

            speed = enemy['speed'] + base_speed_increase

            if dist < speed:
                enemy['pos'] = list(target_pos)
                enemy['path_index'] = 1 - enemy['path_index']  # Flip between 0 and 1
            else:
                enemy['pos'][0] += (dx / dist) * speed
                enemy['pos'][1] += (dy / dist) * speed

            # Update direction vector for vision cone
            next_target = enemy['path'][enemy['path_index']]
            dir_x = next_target[0] - enemy['pos'][0]
            dir_y = next_target[1] - enemy['pos'][1]
            dir_dist = math.sqrt(dir_x ** 2 + dir_y ** 2)
            if dir_dist > 0.01:
                enemy['dir'] = (dir_x / dir_dist, dir_y / dir_dist)

            # Update alert level
            if self._is_in_vision_cone(self.player_pos, enemy):
                enemy['alert_level'] = min(1.0, enemy['alert_level'] + 0.2)
            else:
                enemy['alert_level'] = max(0.0, enemy['alert_level'] - 0.05)

    def _distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _is_valid_and_empty(self, pos):
        x, y = pos
        if not (0 <= x < self.GRID_W and 0 <= y < self.GRID_H):
            return False
        return self.level_grid[y, x] == 0

    def _is_in_vision_cone(self, point, enemy):
        # Convert grid point to pixel coords
        px, py = point
        point_coords = ((px + 0.5) * self.GRID_SIZE, (py + 0.5) * self.GRID_SIZE)

        ex, ey = enemy['pos']
        enemy_center = ((ex + 0.5) * self.GRID_SIZE, (ey + 0.5) * self.GRID_SIZE)

        # Check distance
        dist = self._distance(point_coords, enemy_center)
        if dist > enemy['vision_range'] * self.GRID_SIZE:
            return False

        # Check angle
        vec_to_point = (point_coords[0] - enemy_center[0], point_coords[1] - enemy_center[1])
        angle_to_point = math.atan2(vec_to_point[1], vec_to_point[0])

        enemy_angle = math.atan2(enemy['dir'][1], enemy['dir'][0])

        angle_diff = (angle_to_point - enemy_angle + math.pi) % (2 * math.pi) - math.pi

        vision_angle_rad = math.radians(45)  # Should match render
        return abs(angle_diff) < vision_angle_rad / 2

    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will use a windowed pygame display.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Stealth Horror")

    done = False
    clock = pygame.time.Clock()

    movement = 0
    space = 0
    shift = 0

    print("\n" + "=" * 30)
    print("MANUAL CONTROL MODE")
    print(env.user_guide)
    print("=" * 30 + "\n")

    while not done:
        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Get keyboard input for action ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # Since auto_advance is False, we only step on a key press
        # For manual play, let's step on any action input
        if movement != 0 or space != 0:
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
                # Render final frame
                frame = np.transpose(obs, (1, 0, 2))
                surf = pygame.surfarray.make_surface(frame)
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                # Wait a bit before auto-resetting
                pygame.time.wait(2000)
                obs, info = env.reset()

        # --- Render the observation to the display window ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # To make it playable, we need a small delay so we don't spam actions
        clock.tick(10)

    pygame.quit()