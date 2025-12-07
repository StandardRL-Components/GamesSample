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
        "Controls: ←→ to run, ↑ or Space to jump. Reach the green platform before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, procedurally generated platformer. Race against the clock, making risky jumps to reach the goal. Falling is fatal."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # Persistent state across episodes
    successful_completions = 0
    base_platform_gap = 20
    gap_increase_rate = 5 / 500  # 5 pixels per 500 successful completions

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 40, bold=True)

        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_PLAYER = (255, 64, 64)
        self.COLOR_PLAYER_GLOW = (255, 100, 100)
        self.COLOR_PLATFORM = (64, 128, 255)
        self.COLOR_GOAL = (64, 255, 128)
        self.COLOR_GOAL_GLOW = (128, 255, 180)
        self.COLOR_PARTICLE = (255, 220, 100)
        self.COLOR_UI_TEXT = (220, 220, 240)

        # Game constants
        self.FPS = 30
        self.GRAVITY = 0.8
        self.JUMP_VELOCITY = -13
        self.PLAYER_SPEED = 5.0
        self.PLAYER_SIZE = 20
        self.MAX_STEPS = 1000
        self.MAX_TIME = 20.0

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.platforms = None
        self.goal_rect = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.timer = None
        self.last_x_pos = None

        # Seed the np_random generator for reset
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME

        self._generate_level()

        start_platform = self.platforms[0]
        self.player_pos = np.array([start_platform.centerx, start_platform.top - self.PLAYER_SIZE], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.last_x_pos = self.player_pos[0]

        self.on_ground = True
        self.particles = []

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = []

        # Start platform
        start_plat = pygame.Rect(20, 300, 100, 20)
        self.platforms.append(start_plat)

        current_x = start_plat.right
        current_y = start_plat.y

        gap_increase = GameEnv.successful_completions * GameEnv.gap_increase_rate

        while current_x < self.WIDTH - 150:
            gap = self.np_random.integers(int(GameEnv.base_platform_gap + gap_increase),
                                          int(GameEnv.base_platform_gap + 30 + gap_increase))

            plat_width = self.np_random.integers(80, 150)

            y_change = self.np_random.integers(-70, 70)
            next_y = np.clip(current_y + y_change, 100, self.HEIGHT - 50)

            current_x += gap

            new_plat = pygame.Rect(current_x, next_y, plat_width, 20)
            self.platforms.append(new_plat)

            current_x = new_plat.right
            current_y = new_plat.y

        # Goal platform
        self.goal_rect = pygame.Rect(current_x + 30, current_y, 80, 20)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_held, _ = action

        # Update game logic
        self._update_player(movement, space_held)
        landing_reward = self._handle_collisions()
        self._update_particles()

        self.timer -= 1.0 / self.FPS
        self.steps += 1

        # Calculate rewards
        reward = self._calculate_reward(landing_reward)

        # Check for termination
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.score += reward

        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if not terminated and self.player_pos[0] > self.goal_rect.left:  # Reached goal on timeout
                GameEnv.successful_completions += 1
            elif terminated and self.player_pos[0] > self.goal_rect.left: # Reached goal
                GameEnv.successful_completions += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, movement, space_held):
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel[0] = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel[0] = self.PLAYER_SPEED
        else:  # No horizontal input
            self.player_vel[0] *= 0.8
            if abs(self.player_vel[0]) < 0.1:
                self.player_vel[0] = 0

        # Jumping
        if (movement == 1 or space_held) and self.on_ground:
            self.player_vel[1] = self.JUMP_VELOCITY
            self.on_ground = False
            # sfx: jump_sound()

        # Apply gravity
        self.player_vel[1] += self.GRAVITY
        self.player_vel[1] = min(self.player_vel[1], 15)  # Terminal velocity

        # Update position
        self.player_pos += self.player_vel

        # Screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE / 2, self.WIDTH - self.PLAYER_SIZE / 2)
        if self.player_pos[0] <= self.PLAYER_SIZE / 2 or self.player_pos[0] >= self.WIDTH - self.PLAYER_SIZE / 2:
            self.player_vel[0] = 0

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE / 2, self.player_pos[1], self.PLAYER_SIZE,
                                  self.PLAYER_SIZE)

        was_on_ground = self.on_ground
        self.on_ground = False
        landing_reward = 0

        all_surfaces = self.platforms + [self.goal_rect]

        for plat in all_surfaces:
            if player_rect.colliderect(plat) and self.player_vel[1] > 0:
                # Check if player was above the platform in the previous frame
                prev_player_bottom = player_rect.bottom - self.player_vel[1]
                # FIX: Cast prev_player_bottom to int to avoid floating point inaccuracies
                # that cause the player to phase through platforms.
                if int(prev_player_bottom) <= plat.top:
                    self.player_pos[1] = plat.top - self.PLAYER_SIZE
                    self.player_vel[1] = 0
                    self.on_ground = True
                    if not was_on_ground:
                        landing_reward = 5.0  # Landed on a platform
                        self._create_landing_particles(player_rect.midbottom)
                        # sfx: land_sound()
                    break
        return landing_reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _create_landing_particles(self, pos):
        for _ in range(10):
            angle = self.np_random.uniform(math.pi, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({'pos': np.array(pos, dtype=float), 'vel': vel, 'life': self.np_random.integers(10, 20)})

    def _calculate_reward(self, landing_reward):
        reward = landing_reward

        # Reward for moving towards the goal
        progress = self.player_pos[0] - self.last_x_pos
        reward += progress * 0.1
        self.last_x_pos = self.player_pos[0]

        # Penalty for staying on a platform (encourages speed)
        if self.on_ground:
            reward -= 1.0

        return reward

    def _check_termination(self):
        terminated = False
        terminal_reward = 0

        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE / 2, self.player_pos[1], self.PLAYER_SIZE,
                                  self.PLAYER_SIZE)

        if player_rect.colliderect(self.goal_rect):
            terminated = True
            terminal_reward = 100.0
            # sfx: victory_sound()
        elif self.player_pos[1] > self.HEIGHT:
            terminated = True
            terminal_reward = -100.0
            # sfx: fall_sound()
        elif self.timer <= 0:
            terminated = True

        return terminated, terminal_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, border_radius=3)

        # Draw goal with glow
        glow_rect = self.goal_rect.inflate(10, 10)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.COLOR_GOAL_GLOW, 50), s.get_rect(), border_radius=8)
        self.screen.blit(s, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_GOAL, self.goal_rect, border_radius=3)

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 20.0))
            color = (*self.COLOR_PARTICLE, alpha)
            radius = int(p['life'] / 4)
            if radius > 0:
                s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (radius, radius), radius)
                self.screen.blit(s, (int(p['pos'][0] - radius), int(p['pos'][1] - radius)))

        # Draw player with glow
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = self.player_pos + np.array([0, self.PLAYER_SIZE / 2])

        glow_radius = int(self.PLAYER_SIZE * 0.8)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_PLAYER_GLOW, 100), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (int(player_rect.centerx - glow_radius), int(player_rect.centery - glow_radius)))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_ui(self):
        # Timer
        timer_text = f"TIME: {max(0, self.timer):.1f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (10, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 10))

        # Game Over message
        if self.game_over:
            player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
            player_rect.center = self.player_pos + np.array([0, self.PLAYER_SIZE / 2])

            if player_rect.colliderect(self.goal_rect):
                msg = "LEVEL COMPLETE"
                color = self.COLOR_GOAL
            else:
                msg = "GAME OVER"
                color = self.COLOR_PLAYER

            msg_surf = self.font_msg.render(msg, True, color)
            self.screen.blit(msg_surf, (self.WIDTH / 2 - msg_surf.get_width() / 2, self.HEIGHT / 2 - msg_surf.get_height() / 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "player_pos": self.player_pos.tolist(),
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # To enable rendering, comment out the os.environ line at the top
    # and change the GameEnv constructor to GameEnv(render_mode="human")
    # if you add a "human" render mode.
    
    # For this to run headlessly as is, we need to ensure the main
    # loop doesn't depend on a display.
    is_headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"

    env = GameEnv()
    obs, info = env.reset(seed=42)
    done = False
    
    if not is_headless:
        pygame.display.set_caption("Platformer Environment")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        action = env.action_space.sample() # Default to random action
        
        if not is_headless:
            # Action mapping from keyboard
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            if keys[pygame.K_LEFT]: movement = 3
            if keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

            # Handle quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        if not is_headless:
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if done:
            print(f"Episode finished. Total reward: {total_reward:.2f}")
            print("Resetting environment...")
            if not is_headless:
                pygame.time.wait(2000) # Pause before restarting
            obs, info = env.reset(seed=42)
            total_reward = 0
            done = False # For continuous play
            
        if not is_headless:
            clock.tick(env.FPS)
        
    env.close()