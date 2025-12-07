import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to move. Hold Space to fire at the nearest zombie. Shift does nothing."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of zombies in a top-down horror arena. Last 180 seconds to win. Getting touched by a zombie means game over."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    COLOR_BG = (15, 15, 15)
    COLOR_PLAYER = (0, 255, 127)  # Spring Green
    COLOR_ZOMBIE = (220, 20, 60)  # Crimson
    COLOR_BULLET = (255, 255, 255)
    COLOR_MUZZLE_FLASH = (255, 223, 0)
    COLOR_BLOOD = (139, 0, 0)  # Dark Red
    COLOR_UI_TEXT = (240, 240, 240)

    PLAYER_SIZE = 12
    PLAYER_SPEED = 3

    ZOMBIE_SIZE = 12
    INITIAL_ZOMBIE_COUNT = 50
    INITIAL_ZOMBIE_SPEED = 1.0

    BULLET_SIZE = 3
    BULLET_SPEED = 10
    AMMO_START = 300
    FIRE_RATE_COOLDOWN = 6  # frames (5 shots per second)

    GAME_DURATION_SECONDS = 180
    STAGE_DURATION_SECONDS = 60

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('monospace', 20, bold=True)
        self.font_game_over = pygame.font.SysFont('monospace', 48, bold=True)

        # Initialize state variables to default values to pass validation before reset
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_ammo = self.AMMO_START
        self.zombies = []
        self.bullets = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.fire_cooldown = 0
        self.muzzle_flash_timer = 0
        self.current_stage = 1
        self.current_zombie_speed = self.INITIAL_ZOMBIE_SPEED
        self.np_random = None  # Will be properly initialized in reset()

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_ammo = self.AMMO_START

        self.zombies = []
        for _ in range(self.INITIAL_ZOMBIE_COUNT):
            self._spawn_zombie()

        self.bullets = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.fire_cooldown = 0
        self.muzzle_flash_timer = 0

        self.current_stage = 1
        self.current_zombie_speed = self.INITIAL_ZOMBIE_SPEED

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            # If the game is over, no actions should change the state
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Survival reward per frame

        self._handle_input(action)
        self._update_game_state()

        step_reward = self._handle_collisions_and_rewards()
        reward += step_reward

        self._update_timers_and_stage()

        terminated = self.game_over or self.game_won

        if self.game_won:
            reward += 100
        elif self.game_over:  # Bitten by zombie
            reward += -100

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_zombie(self):
        # Spawn zombies on the edges of the screen
        edge = self.np_random.integers(0, 4)
        if edge == 0:  # top
            pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -self.ZOMBIE_SIZE)
        elif edge == 1:  # bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.ZOMBIE_SIZE)
        elif edge == 2:  # left
            pos = pygame.Vector2(-self.ZOMBIE_SIZE, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        else:  # right
            pos = pygame.Vector2(self.SCREEN_WIDTH + self.ZOMBIE_SIZE, self.np_random.uniform(0, self.SCREEN_HEIGHT))

        self.zombies.append({"pos": pos, "rect": pygame.Rect(pos.x, pos.y, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)})

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Movement
        if movement == 1:  # Up
            self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos.y += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos.x += self.PLAYER_SPEED

        # Clamp player position to screen bounds
        self.player_pos.x = max(0, min(self.player_pos.x, self.SCREEN_WIDTH - self.PLAYER_SIZE))
        self.player_pos.y = max(0, min(self.player_pos.y, self.SCREEN_HEIGHT - self.PLAYER_SIZE))

        # Shooting
        if space_held and self.fire_cooldown == 0 and self.player_ammo > 0:
            self.player_ammo -= 1
            self.fire_cooldown = self.FIRE_RATE_COOLDOWN
            self.muzzle_flash_timer = 2  # frames
            self._fire_bullet()
            # *Play shoot sound*

    def _fire_bullet(self):
        if not self.zombies:
            return  # No zombies to shoot at

        # Find nearest zombie
        player_center = self.player_pos + pygame.Vector2(self.PLAYER_SIZE / 2, self.PLAYER_SIZE / 2)
        closest_zombie = min(self.zombies, key=lambda z: player_center.distance_to(z["pos"]))

        # Calculate direction
        direction = (closest_zombie["pos"] - player_center).normalize()

        # Create bullet
        bullet_start_pos = player_center + direction * (self.PLAYER_SIZE / 2)
        self.bullets.append({"pos": bullet_start_pos, "dir": direction})

    def _update_game_state(self):
        # Update zombies
        for z in self.zombies:
            direction = (self.player_pos - z["pos"])
            if direction.length() > 0:
                direction.normalize_ip()
            z["pos"] += direction * self.current_zombie_speed
            z["rect"].topleft = z["pos"]

        # Update bullets
        for b in self.bullets:
            b["pos"] += b["dir"] * self.BULLET_SPEED

        # Update particles
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _handle_collisions_and_rewards(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Player vs Zombies
        for z in self.zombies:
            if player_rect.colliderect(z["rect"]):
                self.game_over = True
                # *Play player_death sound*
                break
        if self.game_over: return reward  # Stop processing if game is over

        # Bullets vs Zombies
        bullets_to_remove = []
        zombies_to_remove = []
        for i, b in enumerate(self.bullets):
            bullet_rect = pygame.Rect(b["pos"].x, b["pos"].y, self.BULLET_SIZE, self.BULLET_SIZE)

            # Bullet out of bounds
            if not self.screen.get_rect().colliderect(bullet_rect):
                bullets_to_remove.append(i)
                continue

            for j, z in enumerate(self.zombies):
                if j in zombies_to_remove: continue  # Already marked for removal

                if bullet_rect.colliderect(z["rect"]):
                    bullets_to_remove.append(i)
                    zombies_to_remove.append(j)
                    reward += 1.0  # Kill reward
                    self._create_blood_splatter(z["pos"])
                    # *Play zombie_die sound*
                    break  # Bullet can only hit one zombie

        # Remove hit objects
        self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove]

        # To avoid index errors, sort and reverse before popping
        for j in sorted(list(set(zombies_to_remove)), reverse=True):
            self.zombies.pop(j)
            self._spawn_zombie()  # Maintain zombie count

        return reward

    def _create_blood_splatter(self, pos):
        for _ in range(15):
            vel = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2))
            life = self.np_random.integers(10, 20)
            size = self.np_random.integers(2, 4)
            self.particles.append({"pos": pos.copy(), "vel": vel, "life": life, "size": size})

    def _update_timers_and_stage(self):
        # Update cooldowns
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
        if self.muzzle_flash_timer > 0:
            self.muzzle_flash_timer -= 1

        # Check for stage progression
        time_elapsed = self.steps / self.FPS
        if self.current_stage == 1 and time_elapsed >= self.STAGE_DURATION_SECONDS:
            self.current_stage = 2
            self.current_zombie_speed += 0.2
        elif self.current_stage == 2 and time_elapsed >= self.STAGE_DURATION_SECONDS * 2:
            self.current_stage = 3
            self.current_zombie_speed += 0.2

        # Check for win condition
        if time_elapsed >= self.GAME_DURATION_SECONDS:
            self.game_won = True
            self.game_over = True  # End the game

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles (blood)
        for p in self.particles:
            alpha = int(255 * (p["life"] / 20))
            color = self.COLOR_BLOOD + (alpha,)
            pygame.gfxdraw.box(self.screen, (int(p["pos"].x), int(p["pos"].y), p["size"], p["size"]), color)

        # Render zombies
        for z in self.zombies:
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z["rect"])

        # Render player
        player_rect = pygame.Rect(int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        pygame.draw.rect(self.screen, tuple(c / 1.5 for c in self.COLOR_PLAYER), player_rect, 1)  # Border

        # Render muzzle flash
        if self.muzzle_flash_timer > 0:
            player_center = self.player_pos + pygame.Vector2(self.PLAYER_SIZE / 2, self.PLAYER_SIZE / 2)
            pygame.draw.circle(self.screen, self.COLOR_MUZZLE_FLASH, (int(player_center.x), int(player_center.y)), 8)

        # Render bullets
        for b in self.bullets:
            pygame.draw.rect(self.screen, self.COLOR_BULLET,
                             (int(b["pos"].x), int(b["pos"].y), self.BULLET_SIZE, self.BULLET_SIZE))

    def _render_ui(self):
        # Ammo
        ammo_text = self.font_ui.render(f"AMMO: {self.player_ammo}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (10, 10))

        # Timer
        time_elapsed = self.steps / self.FPS
        time_remaining = max(0, self.GAME_DURATION_SECONDS - time_elapsed)
        minutes, seconds = divmod(time_remaining, 60)
        timer_text = self.font_ui.render(f"TIME: {int(minutes):02}:{int(seconds):02}", True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Stage
        stage_text = self.font_ui.render(f"STAGE: {self.current_stage}/3", True, self.COLOR_UI_TEXT)
        stage_rect = stage_text.get_rect(midtop=(self.SCREEN_WIDTH / 2, 10))
        self.screen.blit(stage_text, stage_rect)

        # Game Over / Win message
        if self.game_over and not self.game_won:
            msg = "YOU WERE BITTEN"
            color = self.COLOR_ZOMBIE
        elif self.game_won:
            msg = "YOU SURVIVED"
            color = self.COLOR_PLAYER
        else:
            return

        end_text = self.font_game_over.render(msg, True, color)
        end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ammo": self.player_ammo,
            "stage": self.current_stage,
            "zombies_remaining": len(self.zombies),  # Note: count is constant
        }

    def close(self):
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


if __name__ == "__main__":
    # This block allows you to play the game directly
    # The environment must be created with render_mode="rgb_array"
    # even for human play, as the observation is used for display.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Setup Pygame window for human play
    pygame.display.set_caption("Zombie Arena")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        movement = 0  # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        # Pygame needs (W, H, C) but env gives (H, W, C), so we transpose
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(3000)  # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

    env.close()