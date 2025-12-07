import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:01:17.990208
# Source Brief: brief_00836.md
# Brief Index: 836
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a bouncing ball to hit moving targets.
    
    The player aims to maximize their score by hitting targets and synchronizing hits
    with flashing lights for bonus points. The episode ends if the player loses all
    lives, runs out of time, or reaches the target score.

    **Visuals:**
    - Player: A glowing red ball.
    - Targets: Glowing green squares.
    - Sync Lights: Pulsing yellow rectangles.
    - Background: A dark, futuristic grid.
    - Effects: Particle explosions on target hits.

    **Gameplay:**
    - Use left/right actions to apply horizontal force to the ball.
    - The ball is affected by gravity and bounces off surfaces.
    - Hitting a target earns points.
    - Hitting a target while the sync lights are on earns a large bonus.
    - Missing a target (letting it fall off-screen) costs a life.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a bouncing ball to hit moving targets. "
        "Synchronize hits with flashing lights for bonus points."
    )
    user_guide = "Use the ← and → arrow keys to apply horizontal force to the ball."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    
    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 60)
    COLOR_BALL = (255, 80, 80)
    COLOR_BALL_GLOW = (200, 50, 50)
    COLOR_TARGET = (80, 255, 150)
    COLOR_TARGET_GLOW = (50, 200, 100)
    COLOR_LIGHT_ON = (255, 255, 100)
    COLOR_LIGHT_OFF = (80, 80, 40)
    COLOR_TEXT = (220, 220, 240)
    COLOR_WALL = (180, 180, 200)

    # Game Parameters
    MAX_TIME_SECONDS = 60
    MAX_LIVES = 3
    WIN_SCORE = 500
    GRAVITY = 0.18
    BALL_RADIUS = 12
    HORIZONTAL_ACCEL = 0.4
    BOUNCE_DAMPING = 0.85
    TARGET_SIZE = 24
    MAX_TARGETS = 5
    TARGET_SPAWN_INTERVAL = 1.5 * FPS # Every 1.5 seconds
    LIGHT_PULSE_DURATION = 0.75 * FPS # On for 0.75s, off for 0.75s

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_timer = pygame.font.SysFont('Consolas', 28, bold=True)

        self._initialize_state_variables()

    def _initialize_state_variables(self):
        """Initializes all game state variables. Called by __init__ and reset."""
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.time_remaining = 0
        self.game_over = False

        self.ball_pos = [0.0, 0.0]
        self.ball_vel = [0.0, 0.0]
        
        self.targets = []
        self.particles = []
        
        self.light_on = False
        self.light_timer = 0
        self.target_spawn_timer = 0
        self.base_target_speed = 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.time_remaining = self.MAX_TIME_SECONDS * self.FPS
        self.game_over = False
        
        self.ball_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 4]
        self.ball_vel = [0.0, 0.0]
        
        self.targets = []
        self.particles = []
        
        self.light_on = False
        self.light_timer = self.LIGHT_PULSE_DURATION
        self.target_spawn_timer = self.TARGET_SPAWN_INTERVAL
        self.base_target_speed = 1.0

        for _ in range(3): # Start with a few targets
            self._spawn_target()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1
        
        reward = self._update_game_logic(action)
        terminated = self._check_termination()
        truncated = False # This game does not truncate based on step count

        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            elif self.lives <= 0:
                reward -= 100 # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_game_logic(self, action):
        """Handle input, physics, collisions, and state updates."""
        reward = 0.0

        # 1. Handle Input
        movement = action[0]
        if movement == 3: # Left
            self.ball_vel[0] -= self.HORIZONTAL_ACCEL
        elif movement == 4: # Right
            self.ball_vel[0] += self.HORIZONTAL_ACCEL
        
        # 2. Update Ball Physics
        self.ball_vel[1] += self.GRAVITY
        self.ball_vel[0] *= 0.995 # Air friction
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        # 3. Handle Ball Bounces
        if self.ball_pos[0] < self.BALL_RADIUS:
            self.ball_pos[0] = self.BALL_RADIUS
            self.ball_vel[0] *= -self.BOUNCE_DAMPING
        elif self.ball_pos[0] > self.SCREEN_WIDTH - self.BALL_RADIUS:
            self.ball_pos[0] = self.SCREEN_WIDTH - self.BALL_RADIUS
            self.ball_vel[0] *= -self.BOUNCE_DAMPING

        if self.ball_pos[1] < self.BALL_RADIUS:
            self.ball_pos[1] = self.BALL_RADIUS
            self.ball_vel[1] *= -self.BOUNCE_DAMPING
        elif self.ball_pos[1] > self.SCREEN_HEIGHT - self.BALL_RADIUS:
            self.ball_pos[1] = self.SCREEN_HEIGHT - self.BALL_RADIUS
            self.ball_vel[1] *= -self.BOUNCE_DAMPING

        # 4. Update Targets
        targets_to_remove = []
        for target in self.targets:
            target['rect'].x += target['vel_x']
            if target['rect'].left < 0 or target['rect'].right > self.SCREEN_WIDTH:
                target['vel_x'] *= -1
            
            if target['rect'].top > self.SCREEN_HEIGHT:
                targets_to_remove.append(target)
                self.lives -= 1
                reward -= 10 # Penalty for missing

        self.targets = [t for t in self.targets if t not in targets_to_remove]

        # 5. Check Collisions (Ball <-> Targets)
        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS, 
            self.ball_pos[1] - self.BALL_RADIUS,
            self.BALL_RADIUS * 2, self.BALL_RADIUS * 2
        )
        targets_hit = []
        for target in self.targets:
            if target not in targets_hit and ball_rect.colliderect(target['rect']):
                targets_hit.append(target)
                self.score += 10
                reward += 10

                if self.light_on:
                    self.score += 50
                    reward += 50
                    self._create_particles(target['rect'].center, 40, self.COLOR_LIGHT_ON)
                else:
                    self._create_particles(target['rect'].center, 20, self.COLOR_TARGET)
                
                # Reverse ball velocity for a satisfying bounce-off effect
                self.ball_vel[0] *= -0.5
                self.ball_vel[1] *= -0.8
        
        if targets_hit:
            self.targets = [t for t in self.targets if t not in targets_hit]


        # 6. Update Lights, Particles, and Spawners
        self._update_lights()
        self._update_particles()
        self._update_spawners()
        self._update_difficulty()
        
        return reward

    def _update_lights(self):
        self.light_timer -= 1
        if self.light_timer <= 0:
            self.light_on = not self.light_on
            self.light_timer = self.LIGHT_PULSE_DURATION

    def _update_particles(self):
        particles_to_remove = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                particles_to_remove.append(p)
        self.particles = [p for p in self.particles if p not in particles_to_remove]

    def _update_spawners(self):
        self.target_spawn_timer -= 1
        if self.target_spawn_timer <= 0 and len(self.targets) < self.MAX_TARGETS:
            self._spawn_target()
            self.target_spawn_timer = self.np_random.integers(
                int(self.TARGET_SPAWN_INTERVAL * 0.8),
                int(self.TARGET_SPAWN_INTERVAL * 1.2)
            )


    def _update_difficulty(self):
        # Increase target speed every 10 seconds
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.base_target_speed = min(3.0, self.base_target_speed + 0.2)

    def _spawn_target(self):
        side = self.np_random.choice([-1, 1])
        speed = (self.base_target_speed + self.np_random.uniform(-0.3, 0.3)) * side
        x_pos = -self.TARGET_SIZE if side > 0 else self.SCREEN_WIDTH
        y_pos = self.np_random.uniform(self.SCREEN_HEIGHT * 0.2, self.SCREEN_HEIGHT * 0.6)
        
        target = {
            'rect': pygame.Rect(x_pos, y_pos, self.TARGET_SIZE, self.TARGET_SIZE),
            'vel_x': speed
        }
        self.targets.append(target)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _check_termination(self):
        return self.lives <= 0 or self.time_remaining <= 0 or self.score >= self.WIN_SCORE

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "time_remaining": self.time_remaining / self.FPS,
        }

    def _render_background(self):
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT), 1)
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i), 1)

    def _render_game(self):
        # Render sync lights
        light_color = self.COLOR_LIGHT_ON if self.light_on else self.COLOR_LIGHT_OFF
        light_height = self.SCREEN_HEIGHT // 8
        pygame.draw.rect(self.screen, light_color, (0, 0, self.SCREEN_WIDTH, light_height))
        pygame.draw.rect(self.screen, light_color, (0, self.SCREEN_HEIGHT - light_height, self.SCREEN_WIDTH, light_height))

        # Render walls
        pygame.draw.line(self.screen, self.COLOR_WALL, (0, self.SCREEN_HEIGHT-1), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT-1), 2)

        # Render targets
        for target in self.targets:
            self._draw_glow_rect(target['rect'], self.COLOR_TARGET, self.COLOR_TARGET_GLOW)

        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 30))))
            color = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, color)

        # Render ball
        self._draw_glow_circle(
            (int(self.ball_pos[0]), int(self.ball_pos[1])), 
            self.BALL_RADIUS, self.COLOR_BALL, self.COLOR_BALL_GLOW
        )
    
    def _draw_glow_circle(self, pos, radius, color, glow_color):
        # Draw multiple circles with decreasing alpha for a glow effect
        for i in range(3, 0, -1):
            alpha = 80 // (i * i)
            glow_radius = int(radius * (1 + i * 0.15))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, glow_color + (alpha,))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], glow_radius, glow_color + (alpha,))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

    def _draw_glow_rect(self, rect, color, glow_color):
        inflated_rect = rect.inflate(12, 12)
        glow_surf = pygame.Surface(inflated_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, glow_color + (60,), (0, 0, *inflated_rect.size), border_radius=8)
        self.screen.blit(glow_surf, inflated_rect.topleft)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)


    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 15))

        # Timer
        time_sec = max(0, math.ceil(self.time_remaining / self.FPS))
        timer_color = (255, 100, 100) if time_sec <= 10 else self.COLOR_TEXT
        time_text = self.font_timer.render(f"{time_sec}", True, timer_color)
        self.screen.blit(time_text, (self.SCREEN_WIDTH // 2 - time_text.get_width() // 2, 12))

        # Lives
        lives_text = self.font_main.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 160, 15))
        for i in range(self.lives):
            pos = (self.SCREEN_WIDTH - 75 + i * 25, 27)
            self._draw_glow_circle(pos, 8, self.COLOR_BALL, self.COLOR_BALL_GLOW)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_timer.render(message, True, (255, 255, 255))
            self.screen.blit(end_text, (self.SCREEN_WIDTH // 2 - end_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - 20))


    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # --- Manual Play ---
    # This block will not be run by the evaluation server, but is useful for testing.
    # It requires pygame to be installed and not in headless mode.
    # To run, comment out the `os.environ.setdefault` line at the top of the file.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Check if we are in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Running in headless mode. Manual play is disabled.")
        # Simple test loop
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode finished. Info: {info}")
                obs, info = env.reset()
        env.close()
        exit()

    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bounce Sync")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    while running:
        movement_action = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement_action, space_action, shift_action]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Keep displaying the final frame until R is pressed
            while running:
                final_frame = np.transpose(obs, (1, 0, 2))
                surf = pygame.surfarray.make_surface(final_frame)
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                
                reset_event_found = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        reset_event_found = True
                        break
                if reset_event_found:
                    break
        
        # Convert observation (H, W, C) back to (W, H, C) for pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

    env.close()