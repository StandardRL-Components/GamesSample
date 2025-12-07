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
        "Controls: Press SPACE to dash to the next shadow. Avoid the red guards' line of sight."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A stealth action game. Navigate a noir cityscape, using shadows to hide from patrolling guards as you race to the exit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1500  # Increased for more complex levels

    # Colors (Noir with Neon)
    COLOR_BG = (15, 18, 32)
    COLOR_BUILDING = (25, 30, 45)
    COLOR_SHADOW = (40, 45, 60)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    COLOR_GUARD = (255, 50, 50)
    COLOR_GUARD_GLOW = (255, 100, 100)
    COLOR_VISION_CONE = (180, 50, 50, 100)
    COLOR_EXIT = (50, 255, 150)
    COLOR_EXIT_GLOW = (150, 255, 200)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_PARTICLE = (200, 200, 220)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)

        # All state variables are initialized in reset()
        # self.reset() # This is called by validate_implementation

        # Validate implementation after initialization
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.terminal_reward_given = False
        self.space_pressed_last_frame = False

        # Player state
        self.player_pos = np.array([50.0, self.SCREEN_HEIGHT - 50.0], dtype=np.float64)
        self.player_target_pos = self.player_pos.copy()
        self.player_radius = 8
        self.player_speed = 15.0

        # Level generation
        self._generate_level()

        # Place player in the first shadow
        first_shadow_center = self._get_shadow_center(self.shadows[0])
        self.player_pos = np.array(first_shadow_center, dtype=np.float64)
        self.player_target_pos = self.player_pos.copy()

        # Particles
        self.particles = []

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.shadows = []
        self.guards = []

        # Generate background buildings
        self.buildings = []
        for _ in range(15):
            w = self.np_random.integers(40, 150)
            h = self.np_random.integers(100, self.SCREEN_HEIGHT - 50)
            x = self.np_random.integers(-20, self.SCREEN_WIDTH - w + 20)
            self.buildings.append(pygame.Rect(x, self.SCREEN_HEIGHT - h, w, h))

        # Generate shadows to ensure > 50% coverage and path
        x_pos = self.np_random.integers(10, 40)
        while x_pos < self.SCREEN_WIDTH - 100:
            width = self.np_random.integers(80, 200)
            height = self.np_random.integers(150, self.SCREEN_HEIGHT - 20)
            shadow_rect = pygame.Rect(x_pos, self.SCREEN_HEIGHT - height, width, height)
            self.shadows.append(shadow_rect)
            x_pos += width + self.np_random.integers(0, 60)

        # Exit
        last_shadow_center = self._get_shadow_center(self.shadows[-1])
        self.exit_pos = np.array(last_shadow_center, dtype=np.float64)
        self.exit_radius = 12

        # Guards
        num_guards = self.np_random.integers(3, 6)
        for i in range(num_guards):
            patrol_zone_x = self.np_random.integers(100, self.SCREEN_WIDTH - 100)
            patrol_width = self.np_random.integers(100, 250)
            start_x = max(0, patrol_zone_x - patrol_width // 2)
            end_x = min(self.SCREEN_WIDTH, patrol_zone_x + patrol_width // 2)

            self.guards.append({
                "pos": np.array([float(start_x), float(self.SCREEN_HEIGHT - 30)]),
                "start_x": float(start_x),
                "end_x": float(end_x),
                "direction": 1,
                "speed": self.np_random.uniform(0.8, 1.5),
                "radius": 6,
                "vision_range": self.np_random.integers(100, 150),
                "vision_angle": math.radians(self.np_random.integers(35, 50)),
            })

        # Checkpoints
        self.checkpoints = []
        total_dist = self.exit_pos[0] - self.player_pos[0]
        for i in range(1, 4):
            cp_x = self.player_pos[0] + (total_dist * i / 4)
            self.checkpoints.append({"x": cp_x, "reached": False})

    def step(self, action):
        reward = 0
        if not self.game_over:
            self._handle_input(action)
            self._update_player()
            self._update_guards()
            self._check_detection()
            self._check_exit()

            cp_reward = self._check_checkpoints()
            step_reward = self._calculate_step_reward()
            reward += cp_reward + step_reward

        self._update_particles()

        if self.game_over and not self.terminal_reward_given:
            if self.win_condition:
                reward += 100.0
            else:
                reward += -10.0
            self.terminal_reward_given = True

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:  # Max steps reached
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        space_held = action[1] == 1
        space_just_pressed = space_held and not self.space_pressed_last_frame

        if space_just_pressed:
            # Sound placeholder: # sfx_dash()
            # Find next shadow to the right
            target_shadows_right = [s for s in self.shadows if self._get_shadow_center(s)[0] > self.player_pos[0] + 20]
            if target_shadows_right:
                target_shadows_right.sort(key=lambda s: self._get_shadow_center(s)[0])
                target_shadow = target_shadows_right[0]
            else:
                # If none to right, find nearest to the left
                target_shadows_left = [s for s in self.shadows if self._get_shadow_center(s)[0] < self.player_pos[0] - 20]
                if target_shadows_left:
                    target_shadows_left.sort(key=lambda s: self._get_shadow_center(s)[0], reverse=True)
                    target_shadow = target_shadows_left[0]
                else:  # No valid move
                    return

            self.player_target_pos = np.array(self._get_shadow_center(target_shadow), dtype=np.float64)

        self.space_pressed_last_frame = space_held

    def _update_player(self):
        direction = self.player_target_pos - self.player_pos
        distance = np.linalg.norm(direction)

        if distance > 1:
            self.player_pos += (direction / distance) * min(self.player_speed, distance)
            # Create dash particles
            if self.steps % 2 == 0:
                p_pos = self.player_pos.copy() + self.np_random.uniform(-3, 3, 2)
                p_vel = (direction / distance) * -2 + self.np_random.uniform(-0.5, 0.5, 2)
                self.particles.append({
                    "pos": p_pos, "vel": p_vel, "life": 20, "size": self.np_random.integers(2, 4)
                })

    def _update_guards(self):
        for guard in self.guards:
            guard['pos'][0] += guard['direction'] * guard['speed']
            if guard['pos'][0] >= guard['end_x']:
                guard['pos'][0] = guard['end_x']
                guard['direction'] = -1
            elif guard['pos'][0] <= guard['start_x']:
                guard['pos'][0] = guard['start_x']
                guard['direction'] = 1

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.1)
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _is_player_in_shadow(self):
        player_rect = pygame.Rect(self.player_pos[0] - 1, self.player_pos[1] - 1, 2, 2)
        return any(shadow.colliderect(player_rect) for shadow in self.shadows)

    def _check_detection(self):
        if self._is_player_in_shadow():
            return

        for guard in self.guards:
            to_player = self.player_pos - guard['pos']
            dist_to_player = np.linalg.norm(to_player)

            if 0 < dist_to_player < guard['vision_range']:
                to_player_normalized = to_player / dist_to_player
                guard_facing_vec = np.array([guard['direction'], 0])

                dot_product = np.dot(to_player_normalized, guard_facing_vec)
                angle = math.acos(np.clip(dot_product, -1.0, 1.0))

                if angle < guard['vision_angle'] / 2:
                    # Sound placeholder: # sfx_detected()
                    self.game_over = True
                    self.win_condition = False
                    return

    def _check_exit(self):
        if np.linalg.norm(self.player_pos - self.exit_pos) < self.player_radius + self.exit_radius:
            # Sound placeholder: # sfx_win()
            self.game_over = True
            self.win_condition = True
            self.score += 100  # Evasion score

    def _check_checkpoints(self):
        reward = 0
        for cp in self.checkpoints:
            if not cp["reached"] and self.player_pos[0] >= cp["x"]:
                cp["reached"] = True
                reward += 5.0
                self.score += 5  # Evasion score
        return reward

    def _calculate_step_reward(self):
        if self._is_player_in_shadow():
            return 0.1
        else:
            return -0.2

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background buildings
        for building in self.buildings:
            pygame.draw.rect(self.screen, self.COLOR_BUILDING, building)

        # Shadows
        for shadow in self.shadows:
            pygame.draw.rect(self.screen, self.COLOR_SHADOW, shadow)

        # Exit
        self._draw_glow_circle(self.exit_pos, self.exit_radius, self.COLOR_EXIT, self.COLOR_EXIT_GLOW)

        # Guards and vision cones
        for guard in self.guards:
            self._render_vision_cone(guard)
            self._draw_glow_circle(guard['pos'], guard['radius'], self.COLOR_GUARD, self.COLOR_GUARD_GLOW)

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, p['pos'].astype(int), int(p['size']))

        # Player
        self._draw_glow_circle(self.player_pos, self.player_radius, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

    def _render_ui(self):
        # Score (Evasion)
        score_text = self.font_ui.render(f"Evasion: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Timer (Steps)
        time_text = self.font_ui.render(f"Time: {self.MAX_STEPS - self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (10, 10))

        # Game Over Message
        if self.game_over:
            if self.win_condition:
                end_text = self.font_game_over.render("ESCAPED", True, self.COLOR_EXIT)
            else:
                end_text = self.font_game_over.render("SPOTTED", True, self.COLOR_GUARD)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _render_vision_cone(self, guard):
        pos = guard['pos']
        direction = guard['direction']
        vision_range = guard['vision_range']
        vision_angle = guard['vision_angle']

        p1 = pos
        rad_p2 = math.atan2(0, direction) - vision_angle / 2
        p2 = pos + np.array([math.cos(rad_p2), math.sin(rad_p2)]) * vision_range

        rad_p3 = math.atan2(0, direction) + vision_angle / 2
        p3 = pos + np.array([math.cos(rad_p3), math.sin(rad_p3)]) * vision_range

        points = [p1.astype(int), p2.astype(int), p3.astype(int)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_VISION_CONE)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_VISION_CONE)

    def _draw_glow_circle(self, pos, radius, color, glow_color):
        pos_int = pos.astype(int)
        # Draw multiple layers for glow effect
        for i in range(radius, 0, -2):
            alpha = int(100 * (1 - i / (radius * 2)))
            glow_surface = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (*glow_color, alpha), (radius * 2, radius * 2), i + radius)
            self.screen.blit(glow_surface, (pos_int[0] - radius * 2, pos_int[1] - radius * 2))
        pygame.draw.circle(self.screen, color, pos_int, radius)

    def _get_shadow_center(self, shadow_rect):
        # Return a point low in the shadow, where the player stands
        return (shadow_rect.centerx, shadow_rect.bottom - 30)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos.tolist(),
            "is_in_shadow": self._is_player_in_shadow(),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Beginning implementation validation...")
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)

        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Setup Pygame window for human play
    pygame.display.set_caption("Shadow Dasher")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        action = [0, 0, 0]  # no-op
        # Movement actions are not used in this game design
        if keys[pygame.K_SPACE]:
            action[1] = 1  # Space held
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1  # Shift held (unused)

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            # Wait a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30)  # Run at 30 FPS

    env.close()