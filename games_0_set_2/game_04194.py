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
        "Controls: Use arrow keys to move isometrically. Avoid the red guards' detection cones and reach the blue spaceship."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A futuristic stealth game. Sneak past patrolling alien guards in an isometric facility to reach your escape spaceship."
    )

    # Frames only advance when an action is received.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 25, 20
        self.MAX_STEPS = 1000
        self.MAX_CAUGHT = 3

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_WALL = (80, 90, 110)
        self.COLOR_WALL_TOP = (100, 110, 130)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_GUARD = (255, 50, 50)
        self.COLOR_SPACESHIP = (50, 150, 255)
        self.COLOR_CONE = (255, 80, 80)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_FLASH = (255, 0, 0)

        # Isometric projection
        self.TILE_W_HALF = 16
        self.TILE_H_HALF = 8
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80
        self.WALL_HEIGHT = 20

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 50)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.guards = []
        self.walls = set()
        self.spaceship_pos = None
        self.caught_counter = 0
        self.caught_flash_timer = 0
        self.win_condition = False

        # Initialize state
        # The reset is called here to ensure the environment is ready after __init__
        # This can cause issues if reset() relies on something not yet initialized,
        # but in this case, it's safe.
        self.reset()

    def _generate_level(self):
        self.player_pos = np.array([3, 10])
        self.spaceship_pos = np.array([self.GRID_W - 3, 10])

        self.walls = set()
        for i in range(self.GRID_W):
            self.walls.add((i, 0))
            self.walls.add((i, self.GRID_H - 1))
        for i in range(self.GRID_H):
            self.walls.add((0, i))
            self.walls.add((self.GRID_W - 1, i))

        wall_segments = [
            (range(5, 12), 4), (range(5, 12), 15),
            (11, range(5, 9)), (11, range(11, 15)),
            (16, range(2, 7)), (16, range(13, 18)),
        ]
        for seg in wall_segments:
            if isinstance(seg[0], int):
                for y in seg[1]: self.walls.add((seg[0], y))
            else:
                for x in seg[0]: self.walls.add((x, seg[1]))

        self.guards = [
            {
                "path": [np.array([8, 2]), np.array([8, 17])],
                "cone_range": 6, "wait": 10, "dir": np.array([0, 1.0])
            },
            {
                "path": [np.array([13, 17]), np.array([13, 2])],
                "cone_range": 6, "wait": 10, "dir": np.array([0, -1.0])
            },
            {
                "path": [np.array([19, 6]), np.array([22, 9]), np.array([19, 12]), np.array([17, 10])],
                "cone_range": 5, "wait": 5, "dir": np.array([1.0, 1.0])
            }
        ]
        for guard in self.guards:
            guard["pos"] = guard["path"][0].copy().astype(float)
            guard["path_idx"] = 0
            guard["wait_timer"] = 0
            guard["just_caught"] = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.caught_counter = 0
        self.caught_flash_timer = 0
        self.win_condition = False

        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Small penalty for each step to encourage speed
        self.steps += 1

        if self.caught_flash_timer > 0:
            self.caught_flash_timer -= 1

        # 1. Update Player
        self._move_player(action[0])

        # 2. Update Guards
        self._update_guards()

        # 3. Check for Detection
        for guard in self.guards:
            guard["just_caught"] = False

        if self._check_detection():
            reward -= 10
            self.caught_counter += 1
            self.caught_flash_timer = 5  # Flash for 5 frames

        # 4. Check Termination Conditions
        terminated = False
        if np.array_equal(self.player_pos, self.spaceship_pos):
            reward += 100
            terminated = True
            self.win_condition = True
        elif self.caught_counter >= self.MAX_CAUGHT:
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_player(self, movement):
        direction_map = {
            1: np.array([-1, -1]),  # Up-Left
            2: np.array([1, 1]),    # Down-Right
            3: np.array([-1, 1]),   # Down-Left
            4: np.array([1, -1]),   # Up-Right
        }
        if movement in direction_map:
            move_vec = direction_map[movement]
            new_pos = self.player_pos + move_vec
            if tuple(new_pos) not in self.walls:
                self.player_pos = new_pos

    def _update_guards(self):
        for guard in self.guards:
            if guard["wait_timer"] > 0:
                guard["wait_timer"] -= 1
                continue

            target_pos = guard["path"][guard["path_idx"]]

            if np.linalg.norm(guard["pos"] - target_pos) < 0.1:
                guard["path_idx"] = (guard["path_idx"] + 1) % len(guard["path"])
                guard["wait_timer"] = guard["wait"]
                prev_idx = (guard["path_idx"] - 1 + len(guard["path"])) % len(guard["path"])
                new_target_pos = guard["path"][guard["path_idx"]]
                direction_vec = new_target_pos - guard["path"][prev_idx]
                norm = np.linalg.norm(direction_vec)
                if norm > 0:
                    guard["dir"] = direction_vec / norm
            else:
                direction_vec = target_pos - guard["pos"]
                norm = np.linalg.norm(direction_vec)
                if norm > 0:
                    guard["pos"] += direction_vec / norm * 0.5  # Guard speed

    def _check_detection(self):
        was_caught = False
        for guard in self.guards:
            vec_to_player = self.player_pos - guard["pos"]
            dist_to_player = np.linalg.norm(vec_to_player)

            if 0 < dist_to_player <= guard["cone_range"]:
                player_dir = vec_to_player / dist_to_player
                dot_product = np.dot(player_dir, guard["dir"])

                if dot_product > 0:  # 180 degree cone
                    if self._has_line_of_sight(guard["pos"], self.player_pos):
                        guard["just_caught"] = True
                        was_caught = True
        return was_caught

    def _has_line_of_sight(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        steps = int(max(abs(dx), abs(dy)) * 4)  # Sample more points on the line
        if steps == 0:
            return True

        for i in range(steps + 1):
            t = i / steps
            check_x = int(round(x1 + t * dx))
            check_y = int(round(y1 + t * dy))
            if (check_x, check_y) in self.walls:
                return False
        return True

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_W_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_H_HALF
        return int(screen_x), int(screen_y)

    def _render_iso_poly(self, pos, color, height=0):
        sx, sy = self._iso_to_screen(pos[0], pos[1])
        sy -= height
        points = [
            (sx, sy - self.TILE_H_HALF),
            (sx + self.TILE_W_HALF, sy),
            (sx, sy + self.TILE_H_HALF),
            (sx - self.TILE_W_HALF, sy),
        ]
        pygame.draw.polygon(self.screen, color, points)

    def _render_iso_cube(self, pos, color, top_color, size=1.0):
        x, y = pos[0], pos[1]
        sx, sy = self._iso_to_screen(x, y)

        h_half = self.TILE_H_HALF * size
        w_half = self.TILE_W_HALF * size

        top_points = [
            (sx, sy - h_half),
            (sx + w_half, sy),
            (sx, sy + h_half),
            (sx - w_half, sy)
        ]
        left_points = [
            (sx - w_half, sy),
            (sx, sy + h_half),
            (sx, sy + h_half + self.WALL_HEIGHT),
            (sx - w_half, sy + self.WALL_HEIGHT)
        ]
        right_points = [
            (sx + w_half, sy),
            (sx, sy + h_half),
            (sx, sy + h_half + self.WALL_HEIGHT),
            (sx + w_half, sy + self.WALL_HEIGHT)
        ]

        dark_color_left = tuple(max(0, c - 40) for c in color)
        pygame.draw.polygon(self.screen, dark_color_left, left_points)
        dark_color_right = tuple(max(0, c - 20) for c in color)
        pygame.draw.polygon(self.screen, dark_color_right, right_points)
        pygame.draw.polygon(self.screen, top_color, top_points)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid, spaceship, walls, guards, player in order
        entities = []

        # Add spaceship tile
        entities.append(("poly", self.spaceship_pos, self.COLOR_SPACESHIP, 0))

        # Add walls
        for wall_pos in self.walls:
            entities.append(("cube", wall_pos, self.COLOR_WALL, self.COLOR_WALL_TOP))

        # Add guards and cones
        for guard in self.guards:
            entities.append(("cone", guard, self.COLOR_CONE, 0))
            entities.append(("cube", guard["pos"], self.COLOR_GUARD, self.COLOR_GUARD))

        # Add player
        entities.append(("cube", self.player_pos, self.COLOR_PLAYER, self.COLOR_PLAYER))

        # Sort by isometric y-coordinate for correct occlusion
        # The key now correctly handles the 'cone' entity type, which passes the whole guard dict
        entities.sort(key=lambda e: (e[1]['pos'][0] + e[1]['pos'][1]) if e[0] == 'cone' else (e[1][0] + e[1][1]))

        # Render everything
        self._render_grid()

        for entity in entities:
            e_type, data, color, color2 = entity
            if e_type == "poly":
                self._render_iso_poly(data, color)
            elif e_type == "cube":
                self._render_iso_cube(data, color, color2)
            elif e_type == "cone":
                self._render_detection_cone(data)

        # Pulsing effect for spaceship
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        glow_color = tuple(min(255, c + int(pulse * 50)) for c in self.COLOR_SPACESHIP)
        self._render_iso_poly(self.spaceship_pos, glow_color)

    def _render_grid(self):
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                if (x, y) not in self.walls:
                    self._render_iso_poly((x, y), self.COLOR_GRID)

    def _render_detection_cone(self, guard):
        if self.game_over: return

        guard_pos_screen = self._iso_to_screen(guard["pos"][0], guard["pos"][1])

        cone_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)

        points = [guard_pos_screen]
        angle_rad = math.atan2(guard["dir"][1] * self.TILE_H_HALF - guard["dir"][0] * self.TILE_H_HALF,
                               guard["dir"][0] * self.TILE_W_HALF + guard["dir"][1] * self.TILE_W_HALF)

        for i in range(-90, 91, 10):
            rad = angle_rad + math.radians(i)
            end_x = guard_pos_screen[0] + math.cos(rad) * guard["cone_range"] * self.TILE_W_HALF * 1.5
            end_y = guard_pos_screen[1] + math.sin(rad) * guard["cone_range"] * self.TILE_H_HALF * 1.5
            points.append((int(end_x), int(end_y)))

        alpha = 100 if not guard["just_caught"] else 200
        color = self.COLOR_CONE + (alpha,)
        if len(points) > 2:
            pygame.gfxdraw.filled_polygon(cone_surface, points, color)

        self.screen.blit(cone_surface, (0, 0))

    def _render_ui(self):
        caught_text = self.font_ui.render(f"CAUGHT: {self.caught_counter} / {self.MAX_CAUGHT}", True, self.COLOR_TEXT)
        self.screen.blit(caught_text, (10, 10))

        steps_text = self.font_ui.render(f"STEPS: {self.steps} / {self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        score_text = self.font_ui.render(f"SCORE: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 35))

        if self.caught_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = int(100 * (self.caught_flash_timer / 5))
            flash_surface.fill(self.COLOR_FLASH + (alpha,))
            self.screen.blit(flash_surface, (0, 0))

        if self.game_over:
            msg = "MISSION COMPLETE" if self.win_condition else "GAME OVER"
            color = self.COLOR_PLAYER if self.win_condition else self.COLOR_GUARD

            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_big.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "caught_counter": self.caught_counter,
            "player_pos": self.player_pos.tolist(),
            "win": self.win_condition,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows for human play and testing
    # It will not be run by the evaluation system
    # but is useful for development.
    
    # Un-set the headless driver for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Stealth Game")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    print("\n" + "=" * 30)
    print(env.game_description)
    print(env.user_guide)
    print("=" * 30 + "\n")

    while running:
        movement = 0  # No-op

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reset
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")
                if event.key == pygame.K_q:
                    running = False

        # Map keyboard to MultiDiscrete action
        # This is a simplified mapping for human play.
        # Direction map: 1:Up-Left, 2:Down-Right, 3:Down-Left, 4:Up-Right
        # On-screen, this corresponds to: 1:Up, 2:Down, 3:Left, 4:Right
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        # The other actions are not used for human play
        action = [movement, 0, 0]

        if movement != 0 or env.auto_advance:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        print("--- Game Reset ---")
                        wait_for_reset = False

        clock.tick(15)  # Control game speed for human play

    env.close()