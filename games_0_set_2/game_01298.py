import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys to move. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, retro top-down shooter. Destroy all alien invaders across three stages to win. You have three lives."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
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
        self.PLAYER_SPEED = 7
        self.PROJECTILE_SPEED = 12
        self.ALIEN_PROJECTILE_SPEED = 4
        self.INITIAL_ALIEN_SPEED = 1.0
        self.FIRE_COOLDOWN_FRAMES = 8
        self.MAX_STEPS = 1000
        self.TOTAL_ALIENS = 50
        self.PLAYER_SIZE = 12
        self.ALIEN_SIZE = 16
        self.BORDER = 20

        # --- Colors ---
        self.COLOR_BG = (10, 15, 25)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_ALIEN_PROJECTILE = (255, 150, 50)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_BORDER = (80, 80, 100)
        self.COLOR_EXPLOSION = (255, 255, 255)

        # --- Fonts ---
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_stage = pygame.font.SysFont("monospace", 22, bold=True)

        # Initialize state variables to a valid, non-None state for validation
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.player_lives = 3
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.aliens_killed = 0
        self.current_stage = 1
        self.alien_speed = self.INITIAL_ALIEN_SPEED
        self.fire_cooldown = 0
        self.prev_space_held = False
        self.aliens = []
        self.projectiles = []
        self.alien_projectiles = []
        self.explosions = []

        # This will be properly initialized in reset()
        self.np_random = None

        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.player_lives = 3
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.aliens_killed = 0
        self.current_stage = 1
        self.alien_speed = self.INITIAL_ALIEN_SPEED
        self.fire_cooldown = 0
        self.prev_space_held = False

        self.projectiles = []
        self.alien_projectiles = []
        self.explosions = []

        self._spawn_aliens()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for each step to encourage speed

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1

            self._handle_input(movement, space_held)

            # Update game logic
            event_rewards = self._update_game_state()
            reward += event_rewards

            # Update cooldowns and counters
            self.steps += 1
            if self.fire_cooldown > 0:
                self.fire_cooldown -= 1

            # Check for stage progression
            self._update_stage()

        # Check for termination conditions
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.player_lives <= 0:
                reward -= 100
            elif self.aliens_killed >= self.TOTAL_ALIENS:
                reward += 100

        if self.auto_advance:
            self.clock.tick(30)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _handle_input(self, movement, space_held):
        # Player Movement
        if movement == 1:
            self.player_pos[1] -= self.PLAYER_SPEED  # Up
        if movement == 2:
            self.player_pos[1] += self.PLAYER_SPEED  # Down
        if movement == 3:
            self.player_pos[0] -= self.PLAYER_SPEED  # Left
        if movement == 4:
            self.player_pos[0] += self.PLAYER_SPEED  # Right

        # Clamp player position to within game boundaries
        self.player_pos[0] = np.clip(
            self.player_pos[0], self.BORDER, self.WIDTH - self.BORDER
        )
        self.player_pos[1] = np.clip(
            self.player_pos[1], self.BORDER, self.HEIGHT - self.BORDER
        )

        # Player Firing
        # Fire on press (transition from not held to held)
        if space_held and not self.prev_space_held and self.fire_cooldown == 0:
            # Sfx: player_shoot.wav
            self.projectiles.append(list(self.player_pos))
            self.fire_cooldown = self.FIRE_COOLDOWN_FRAMES
        self.prev_space_held = space_held

    def _update_game_state(self):
        total_reward = 0

        total_reward += self._update_projectiles()
        self._update_aliens()
        total_reward += self._update_alien_projectiles()
        self._update_explosions()

        return total_reward

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            proj[1] -= self.PROJECTILE_SPEED
            if proj[1] < 0:
                self.projectiles.remove(proj)
                continue

            proj_rect = pygame.Rect(proj[0] - 2, proj[1] - 2, 4, 4)
            for alien in self.aliens[:]:
                alien_rect = pygame.Rect(
                    alien["pos"][0] - self.ALIEN_SIZE / 2,
                    alien["pos"][1] - self.ALIEN_SIZE / 2,
                    self.ALIEN_SIZE,
                    self.ALIEN_SIZE,
                )
                if proj_rect.colliderect(alien_rect):
                    # Sfx: alien_explosion.wav
                    self.aliens.remove(alien)
                    self.projectiles.remove(proj)
                    self.explosions.append(
                        {"pos": alien["pos"][:], "radius": self.ALIEN_SIZE, "life": 20}
                    )
                    self.score += 10
                    reward += 10  # Main event reward
                    reward += 0.1  # Continuous feedback reward
                    self.aliens_killed += 1
                    break
        return reward

    def _update_aliens(self):
        for i, alien in enumerate(self.aliens):
            # Stage-based movement patterns
            if self.current_stage == 1:  # Horizontal
                alien["pos"][0] += alien["vel"][0]
                if not (self.BORDER < alien["pos"][0] < self.WIDTH - self.BORDER):
                    alien["vel"][0] *= -1

            elif self.current_stage == 2:  # Diagonal Bounce
                alien["pos"][0] += alien["vel"][0]
                alien["pos"][1] += alien["vel"][1]
                if not (self.BORDER < alien["pos"][0] < self.WIDTH - self.BORDER):
                    alien["vel"][0] *= -1
                if not (
                    self.BORDER < alien["pos"][1] < self.HEIGHT - self.BORDER - 100
                ):
                    alien["vel"][1] *= -1

            elif self.current_stage == 3:  # Circular
                alien["angle"] += alien["vel"][0] * 0.05
                alien["pos"][0] = (
                    alien["center"][0] + math.cos(alien["angle"]) * alien["radius"]
                )
                alien["pos"][1] = (
                    alien["center"][1] + math.sin(alien["angle"]) * alien["radius"]
                )

            # Alien Firing
            if self.steps > 0 and (self.steps + i * 7) % 50 == 0:
                # Sfx: alien_shoot.wav
                self.alien_projectiles.append(alien["pos"][:])

    def _update_alien_projectiles(self):
        reward = 0
        player_rect = pygame.Rect(
            self.player_pos[0] - self.PLAYER_SIZE / 2,
            self.player_pos[1] - self.PLAYER_SIZE / 2,
            self.PLAYER_SIZE,
            self.PLAYER_SIZE,
        )

        for proj in self.alien_projectiles[:]:
            proj[1] += self.ALIEN_PROJECTILE_SPEED
            if proj[1] > self.HEIGHT:
                self.alien_projectiles.remove(proj)
                continue

            proj_rect = pygame.Rect(proj[0] - 2, proj[1] - 2, 4, 4)
            if player_rect.colliderect(proj_rect):
                # Sfx: player_hit.wav
                self.alien_projectiles.remove(proj)
                self.player_lives -= 1
                self.score = max(0, self.score - 5)
                reward -= 5
                self.explosions.append(
                    {
                        "pos": self.player_pos[:],
                        "radius": self.PLAYER_SIZE * 1.5,
                        "life": 25,
                    }
                )
                # Respawn player in center
                self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50]
                break  # Player can only be hit once per frame
        return reward

    def _update_explosions(self):
        for exp in self.explosions[:]:
            exp["life"] -= 1
            exp["radius"] += 1
            if exp["life"] <= 0:
                self.explosions.remove(exp)

    def _spawn_aliens(self):
        self.aliens = []
        rows = 5
        cols = self.TOTAL_ALIENS // rows
        for i in range(self.TOTAL_ALIENS):
            row = i // cols
            col = i % cols
            x = self.BORDER + 30 + (self.WIDTH - self.BORDER * 2 - 60) * (
                col / (cols - 1)
            )
            y = self.BORDER + 30 + row * 30

            alien = {
                "pos": [x, y],
                "vel": [
                    self.alien_speed * self.np_random.choice([-1, 1]),
                    self.alien_speed,
                ],
                "center": [x, y + 20],  # For circular motion
                "radius": 20,  # For circular motion
                "angle": self.np_random.uniform(
                    0, 2 * math.pi
                ),  # For circular motion
            }
            self.aliens.append(alien)

    def _update_stage(self):
        new_stage = self.current_stage
        if self.aliens_killed >= 17 and self.current_stage == 1:
            new_stage = 2
        if self.aliens_killed >= 34 and self.current_stage == 2:
            new_stage = 3

        if new_stage != self.current_stage:
            self.current_stage = new_stage
            self.alien_speed += 0.1
            # Update velocities for all remaining aliens
            for alien in self.aliens:
                dir_x = (
                    np.sign(alien["vel"][0])
                    if alien["vel"][0] != 0
                    else self.np_random.choice([-1, 1])
                )
                dir_y = np.sign(alien["vel"][1]) if alien["vel"][1] != 0 else 1
                alien["vel"] = [self.alien_speed * dir_x, self.alien_speed * dir_y]

    def _check_termination(self):
        return (
            self.player_lives <= 0
            or self.aliens_killed >= self.TOTAL_ALIENS
            or self.steps >= self.MAX_STEPS
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw borders
        pygame.draw.rect(
            self.screen,
            self.COLOR_BORDER,
            (self.BORDER, self.BORDER, self.WIDTH - 2 * self.BORDER, self.HEIGHT - 2 * self.BORDER),
            2,
        )

        # Draw aliens
        for alien in self.aliens:
            pos = (int(alien["pos"][0]), int(alien["pos"][1]))
            size = self.ALIEN_SIZE
            pygame.draw.rect(
                self.screen,
                self.COLOR_ALIEN,
                (pos[0] - size / 2, pos[1] - size / 2, size, size),
            )

        # Draw projectiles
        for proj in self.projectiles:
            pygame.draw.circle(
                self.screen, self.COLOR_PROJECTILE, (int(proj[0]), int(proj[1])), 3
            )
        for proj in self.alien_projectiles:
            pygame.draw.circle(
                self.screen, self.COLOR_ALIEN_PROJECTILE, (int(proj[0]), int(proj[1])), 4
            )

        # Draw explosions
        for exp in self.explosions:
            alpha = max(0, 255 * (exp["life"] / 25))
            pos = (int(exp["pos"][0]), int(exp["pos"][1]))
            radius = int(exp["radius"])
            if radius > 0:
                pygame.gfxdraw.aacircle(
                    self.screen, pos[0], pos[1], radius, (*self.COLOR_EXPLOSION, alpha)
                )
                pygame.gfxdraw.filled_circle(
                    self.screen, pos[0], pos[1], radius, (*self.COLOR_EXPLOSION, alpha / 4)
                )

        # Draw player
        if self.player_lives > 0:
            p = self.player_pos
            s = self.PLAYER_SIZE
            points = [(p[0], p[1] - s), (p[0] - s / 2, p[1] + s / 2), (p[0] + s / 2, p[1] + s / 2)]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (self.BORDER + 10, self.BORDER - 20))

        # Lives
        lives_text = self.font_ui.render(
            f"LIVES: {max(0, self.player_lives)}", True, self.COLOR_UI
        )
        self.screen.blit(
            lives_text,
            (self.WIDTH - self.BORDER - lives_text.get_width() - 10, self.BORDER - 20),
        )

        # Stage
        stage_text_str = (
            "VICTORY!"
            if self.aliens_killed >= self.TOTAL_ALIENS
            else f"STAGE {self.current_stage}"
        )
        if self.player_lives <= 0:
            stage_text_str = "GAME OVER"
        stage_text = self.font_stage.render(stage_text_str, True, self.COLOR_UI)
        self.screen.blit(
            stage_text,
            (
                self.WIDTH / 2 - stage_text.get_width() / 2,
                self.HEIGHT - self.BORDER - 5,
            ),
        )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "aliens_remaining": self.TOTAL_ALIENS - self.aliens_killed,
            "stage": self.current_stage,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        # We need a temporary RNG for validation before the first real reset
        temp_seed = 12345
        _ = self.reset(seed=temp_seed)
        obs, info = self.reset(seed=temp_seed)
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()

    # Create a window to display the game
    pygame.display.set_caption("Gymnasium Game")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    terminated = False

    # Mapping from Pygame keys to your environment's actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    print(env.user_guide)
    print("Close the window to quit.")

    while not terminated:
        # --- Human Input ---
        movement_action = 0  # No-op by default
        space_action = 0
        shift_action = 0

        keys = pygame.key.get_pressed()
        for key, action_val in key_to_action.items():
            if keys[key]:
                movement_action = action_val
                break  # Prioritize first key found (e.g., up over down)

        if keys[pygame.K_SPACE]:
            space_action = 1

        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

    print(f"Game Over! Final Info: {info}")
    env.close()