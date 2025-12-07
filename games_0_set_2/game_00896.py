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
        "Controls: Arrow keys to jump directionally. Hold Shift for nearest platform, "
        "or Space for furthest. Reach platform 50 to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop across procedurally generated number platforms to reach the target number. "
        "Jumping further is riskier but more rewarding. Don't fall!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Visuals & Theming
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 10)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 200)
        self.COLOR_SAFE = (20, 200, 120)  # Green
        self.COLOR_RISKY = (255, 80, 80)  # Red
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TARGET_PLATFORM = (255, 215, 0)  # Gold

        self.font_main = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_platform = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_player = pygame.font.SysFont("monospace", 14, bold=True)

        # Game constants
        self.TARGET_NUMBER = 50
        self.MAX_EPISODE_STEPS = 10000
        self.NUM_COLUMNS = 8
        self.PLATFORM_SIZE = 60
        self.JUMP_DURATION = 20  # frames
        self.JUMP_HEIGHT = 50
        self.MAX_JUMP_RADIUS = 250

        # State variables will be initialized in reset()
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.win = False

        self.platform_speed = 0.0
        self.gap_probability = 0.0
        self.highest_num_generated = 0
        self.platform_id_counter = 0

        self.player = {}
        self.platforms = []
        self.particles = []
        self.current_reward = 0.0

        # Initialize state
        # self.reset() is not called in __init__ to allow for seeding before first reset
        # Run self-check
        # self.validate_implementation() # Commented out as it relies on reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.win = False

        self.platform_speed = 1.0
        self.gap_probability = 0.1
        self.highest_num_generated = 0
        self.platform_id_counter = 0

        self.particles = []
        self.player = {}
        self.platforms = []

        self._generate_initial_platforms()
        self._place_player_on_start()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.current_reward = 0.0

        self._update_difficulty()
        self._update_platforms()
        self._handle_player_movement(action)
        self._update_particles()
        self._cleanup_objects()

        terminated = self._check_termination()

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            self.current_reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 500 == 0:
            self.platform_speed = min(3.0, self.platform_speed + 0.05)
        if self.steps > 0 and self.steps % 250 == 0:
            self.gap_probability = min(0.5, self.gap_probability + 0.01)

    def _handle_player_movement(self, action):
        if self.player["is_jumping"]:
            self._update_jump()
        else:
            self._follow_platform()
            if not self.game_over:
                self._process_action(action)
                # Idle penalty
                if not self.player["is_jumping"]:
                    self.current_reward -= 0.02

    def _process_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        current_platform = self._get_player_platform()
        if not current_platform:
            return

        other_platforms = [
            p for p in self.platforms if p["id"] != current_platform["id"]
        ]
        if not other_platforms:
            return

        target_platform = None

        if space_held:  # Furthest jump
            valid_targets = [
                p
                for p in other_platforms
                if math.hypot(
                    p["rect"].centerx - self.player["x"],
                    p["rect"].centery - self.player["y"],
                )
                <= self.MAX_JUMP_RADIUS
            ]
            if valid_targets:
                target_platform = max(
                    valid_targets,
                    key=lambda p: math.hypot(
                        p["rect"].centerx - self.player["x"],
                        p["rect"].centery - self.player["y"],
                    ),
                )
        elif shift_held:  # Nearest jump
            target_platform = min(
                other_platforms,
                key=lambda p: math.hypot(
                    p["rect"].centerx - self.player["x"],
                    p["rect"].centery - self.player["y"],
                ),
            )
        elif movement > 0:  # Directional jump
            px, py = self.player["x"], self.player["y"]

            if movement == 1:  # Up
                candidates = [p for p in other_platforms if p["rect"].centery < py]
            elif movement == 2:  # Down
                candidates = [p for p in other_platforms if p["rect"].centery > py]
            elif movement == 3:  # Left
                candidates = [p for p in other_platforms if p["rect"].centerx < px]
            else:  # Right (4)
                candidates = [p for p in other_platforms if p["rect"].centerx > px]

            if candidates:
                # Find closest in the chosen direction
                target_platform = min(
                    candidates,
                    key=lambda p: math.hypot(p["rect"].centerx - px, p["rect"].centery - py),
                )

        if target_platform:
            self._initiate_jump(target_platform)

    def _initiate_jump(self, target_platform):
        start_pos = (self.player["x"], self.player["y"])
        end_pos = target_platform["rect"].center

        self.player.update(
            {
                "is_jumping": True,
                "jump_progress": 0,
                "jump_start_pos": start_pos,
                "jump_end_pos": end_pos,
                "jump_target_id": target_platform["id"],
            }
        )

        # Risk/reward for jump distance
        dist = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
        if dist > self.PLATFORM_SIZE * 2:
            self.current_reward += 1.0  # Risky jump bonus
        else:
            self.current_reward -= 1.0  # Safe jump penalty

    def _update_jump(self):
        self.player["jump_progress"] += 1 / self.JUMP_DURATION
        progress = self.player["jump_progress"]

        if progress >= 1.0:
            # Land
            self.player["is_jumping"] = False
            target_platform = next(
                (p for p in self.platforms if p["id"] == self.player["jump_target_id"]),
                None,
            )

            if target_platform:
                self.player["x"], self.player["y"] = target_platform["rect"].center
                self.player["on_platform_id"] = target_platform["id"]
                self.score += target_platform["number"]
                self.current_reward += 0.1  # Landing reward
                self._create_particles(self.player["x"], self.player["y"], 20)
                # sfx: land

                if target_platform["number"] == self.TARGET_NUMBER:
                    self.win = True
                    self.game_over = True
                    self.current_reward += 100  # Win bonus
                    # sfx: win_fanfare
            else:
                # Fell into a gap
                self.player["x"], self.player["y"] = self.player["jump_end_pos"]
                self._fall()
        else:
            # Interpolate position for smooth arc
            start_x, start_y = self.player["jump_start_pos"]
            end_x, end_y = self.player["jump_end_pos"]
            self.player["x"] = start_x + (end_x - start_x) * progress
            self.player["y"] = start_y + (end_y - start_y) * progress
            # Add arc
            arc = math.sin(progress * math.pi) * self.JUMP_HEIGHT
            self.player["y"] -= arc

    def _fall(self):
        self.lives -= 1
        self.current_reward -= 100  # Fall penalty
        self.game_over = self.lives <= 0
        # sfx: fall
        if not self.game_over:
            # Respawn on the last safe platform
            last_platform = self._get_player_platform()
            if last_platform:
                self.player["x"], self.player["y"] = last_platform["rect"].center
            else:  # Failsafe if last platform is gone
                self._place_player_on_start()

    def _follow_platform(self):
        platform = self._get_player_platform()
        if platform:
            self.player["y"] = platform["rect"].centery
            if platform["rect"].top < 0:  # Scrolled off screen
                self._fall()
        else:
            # Player was on a platform that got removed
            self._fall()

    def _update_platforms(self):
        for p in self.platforms:
            p["y_float"] -= self.platform_speed
            p["rect"].y = int(p["y_float"])

    def _update_particles(self):
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _cleanup_objects(self):
        self.platforms = [p for p in self.platforms if p["rect"].bottom > 0]
        self._generate_new_platforms()

    def _generate_initial_platforms(self):
        self.platforms = []
        self.platform_id_counter = 0

        # Start platform
        start_x = self.screen_width / 2
        start_y = self.screen_height * 0.75
        self._create_platform(start_x, start_y, 1)

        # Populate screen
        for y in range(
            int(start_y),
            self.screen_height + self.PLATFORM_SIZE,
            int(self.PLATFORM_SIZE * 1.5),
        ):
            self._generate_platform_row(y)

    def _generate_new_platforms(self):
        if not self.platforms or max(p["rect"].top for p in self.platforms) < self.screen_height:
            last_y = (
                max(p["rect"].bottom for p in self.platforms)
                if self.platforms
                else self.screen_height
            )
            new_y = last_y + self.PLATFORM_SIZE * 1.5
            self._generate_platform_row(int(new_y))

    def _generate_platform_row(self, y_pos):
        col_width = self.screen_width / self.NUM_COLUMNS
        
        # FIX: Check if player has been placed before trying to access its platform
        if 'on_platform_id' in self.player:
            player_platform = self._get_player_platform()
            current_player_num = player_platform['number'] if player_platform else 1
        else:
            # During initial setup, the player isn't placed yet.
            # The "current" platform is assumed to be the starting one, which has number 1.
            current_player_num = 1

        # Chance to spawn the target platform if conditions are met
        target_spawned = any(p["number"] == self.TARGET_NUMBER for p in self.platforms)
        if (
            not target_spawned
            and current_player_num > self.TARGET_NUMBER - 10
            and self.np_random.random() < 0.2
        ):
            col = self.np_random.integers(0, self.NUM_COLUMNS)
            x_pos = col * col_width + col_width / 2
            self._create_platform(x_pos, y_pos, self.TARGET_NUMBER)
            return  # Only one target platform

        for i in range(self.NUM_COLUMNS):
            if self.np_random.random() > self.gap_probability:
                x_pos = i * col_width + col_width / 2

                num_min = max(1, current_player_num - 2)
                num_max = min(self.TARGET_NUMBER - 1, self.highest_num_generated + 3)
                if num_max <= num_min:
                    num_max = num_min + 2

                num = self.np_random.integers(num_min, num_max + 1)
                self._create_platform(x_pos, y_pos, num)

    def _create_platform(self, x, y, num):
        rect = pygame.Rect(0, 0, self.PLATFORM_SIZE, self.PLATFORM_SIZE)
        rect.center = (int(x), int(y))

        self.platforms.append(
            {"id": self.platform_id_counter, "rect": rect, "y_float": float(y), "number": num}
        )
        self.platform_id_counter += 1
        self.highest_num_generated = max(self.highest_num_generated, num)

    def _place_player_on_start(self):
        start_platform = self.platforms[0]
        self.player = {
            "x": start_platform.get("rect").centerx,
            "y": start_platform.get("rect").centery,
            "on_platform_id": start_platform.get("id"),
            "is_jumping": False,
            "jump_progress": 0,
            "jump_start_pos": (0, 0),
            "jump_end_pos": (0, 0),
            "jump_target_id": -1,
        }

    def _get_player_platform(self):
        if 'on_platform_id' not in self.player:
            return None
        return next(
            (p for p in self.platforms if p["id"] == self.player["on_platform_id"]), None
        )

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.screen_height):
            interp = y / self.screen_height
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y))

    def _render_game(self):
        # Render platforms
        player_pos = (self.player["x"], self.player["y"])
        for p in self.platforms:
            dist = math.hypot(
                p["rect"].centerx - player_pos[0], p["rect"].centery - player_pos[1]
            )
            risk_factor = min(1.0, dist / self.MAX_JUMP_RADIUS)

            if p["number"] == self.TARGET_NUMBER:
                color = self.COLOR_TARGET_PLATFORM
            else:
                color = (
                    int(
                        self.COLOR_SAFE[0] * (1 - risk_factor)
                        + self.COLOR_RISKY[0] * risk_factor
                    ),
                    int(
                        self.COLOR_SAFE[1] * (1 - risk_factor)
                        + self.COLOR_RISKY[1] * risk_factor
                    ),
                    int(
                        self.COLOR_SAFE[2] * (1 - risk_factor)
                        + self.COLOR_RISKY[2] * risk_factor
                    ),
                )

            pygame.draw.rect(self.screen, color, p["rect"], border_radius=5)
            num_surf = self.font_platform.render(str(p["number"]), True, self.COLOR_TEXT)
            self.screen.blit(num_surf, num_surf.get_rect(center=p["rect"].center))

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(p["x"]), int(p["y"]), int(p["size"]), color)
            except TypeError: # Handle potential color tuple issue
                pygame.gfxdraw.filled_circle(self.screen, int(p["x"]), int(p["y"]), int(p["size"]), (*p["color"],))


        # Render player
        player_x, player_y = int(self.player["x"]), int(self.player["y"])
        glow_radius = 15
        player_radius = 10

        # Glow effect
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            glow_surf, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius
        )
        self.screen.blit(glow_surf, (player_x - glow_radius, player_y - glow_radius))

        pygame.gfxdraw.aacircle(
            self.screen, player_x, player_y, player_radius, self.COLOR_PLAYER
        )
        pygame.gfxdraw.filled_circle(
            self.screen, player_x, player_y, player_radius, self.COLOR_PLAYER
        )

    def _render_ui(self):
        # Score and Lives
        score_text = f"SCORE: {self.score}"
        lives_text = f"LIVES: {self.lives}"
        target_text = f"TARGET: {self.TARGET_NUMBER}"

        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        lives_surf = self.font_main.render(lives_text, True, self.COLOR_TEXT)
        target_surf = self.font_main.render(target_text, True, self.COLOR_TEXT)

        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(lives_surf, (10, 30))
        self.screen.blit(
            target_surf, target_surf.get_rect(centerx=self.screen_width / 2, y=10)
        )

        # Current platform number above player
        if not self.player["is_jumping"]:
            platform = self._get_player_platform()
            if platform:
                num_surf = self.font_player.render(
                    str(platform["number"]), True, self.COLOR_TEXT
                )
                self.screen.blit(
                    num_surf,
                    num_surf.get_rect(center=(self.player["x"], self.player["y"] - 25)),
                )

        if self.game_over:
            overlay = pygame.Surface(
                (self.screen_width, self.screen_height), pygame.SRCALPHA
            )
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            end_text = "YOU WIN!" if self.win else "GAME OVER"
            end_surf = pygame.font.SysFont("monospace", 50, bold=True).render(
                end_text, True, self.COLOR_TEXT
            )
            self.screen.blit(
                end_surf,
                end_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2)),
            )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_EPISODE_STEPS

    def _create_particles(self, x, y, count):
        # sfx: particle_burst
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            self.particles.append(
                {
                    "x": x,
                    "y": y,
                    "vx": math.cos(angle) * speed,
                    "vy": math.sin(angle) * speed,
                    "life": self.np_random.integers(15, 30),
                    "max_life": 30,
                    "size": self.np_random.random() * 2 + 1,
                    "color": self.COLOR_SAFE,
                }
            )

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # Run validation
    try:
        env.validate_implementation()
    except Exception as e:
        print(f"Validation failed: {e}")


    obs, info = env.reset()
    done = False

    # For human play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.display.init()
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Number Hopper")

    # Action state
    movement = 0
    space_held = 0
    shift_held = 0

    print(GameEnv.user_guide)

    while not done:
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

            # Key down
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space_held = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 1
                elif event.key == pygame.K_r:  # Reset
                    obs, info = env.reset()

            # Key up
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    movement = 0
                elif event.key == pygame.K_SPACE:
                    space_held = 0
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 0

        action = [movement, space_held, shift_held]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # Render to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # In auto_advance=True, the env's internal clock handles the frame rate.
        # We just need to keep the loop running.
        env.clock.tick(30)  # Match the internal clock for smooth display

    env.close()
    print("Game Over!")
    print(f"Final Score: {info['score']}")