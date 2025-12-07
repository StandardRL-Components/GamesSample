import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:59:11.062345
# Source Brief: brief_00739.md
# Brief Index: 739
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a bouncing ball.
    The goal is to hit targets to score points, while managing a shield
    to avoid penalties from hitting walls. The episode ends when the
    player reaches the target score or time runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a bouncing ball to hit targets for points. Use a shield to avoid penalties "
        "when hitting walls, but be aware that activating it costs points."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to apply force to the ball. Press space to activate your shield."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 60 * FPS  # 60 seconds
    WIN_SCORE = 100

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_BALL = (255, 255, 255)
    COLOR_TARGET = (0, 255, 150)
    COLOR_WALL_IDLE = (80, 90, 110)
    COLOR_SHIELD = (50, 150, 255)
    COLOR_TEXT = (220, 220, 230)
    COLOR_SCORE = (255, 215, 0)
    COLOR_TIMER = (255, 80, 80)

    # Game Mechanics
    BALL_RADIUS = 12
    TARGET_RADIUS = 18
    NUM_TARGETS = 3
    BALL_ACCEL = 0.5
    BALL_DAMPING = 0.998
    MAX_VEL = 15
    SHIELD_COST = 10
    SHIELD_DURATION_STEPS = 2 * FPS  # 2 seconds

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_subtitle = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 30)
            self.font_subtitle = pygame.font.Font(None, 24)

        # Initialize state variables
        self.ball_pos = None
        self.ball_vel = None
        self.targets = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.shield_active = False
        self.shield_timer = 0
        self.prev_space_held = False
        self.wall_hit_flash = 0
        
        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Ball state
        self.ball_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.ball_vel = np.array([0.0, 0.0], dtype=np.float32)

        # Shield state
        self.shield_active = False
        self.shield_timer = 0
        self.prev_space_held = False
        
        # Targets and particles
        self.targets = []
        for _ in range(self.NUM_TARGETS):
            self._spawn_target()
        self.particles = []

        self.wall_hit_flash = 0

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game Logic ---
        self._update_shield()
        reward += self._update_ball()
        reward += self._update_targets()
        self._update_particles()
        
        # --- Check Termination ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            reward -= 50.0
            terminated = True
            self.game_over = True

        # Ensure score doesn't go below zero
        self.score = max(0, self.score)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held_raw, _ = action
        space_held = space_held_raw == 1

        # Movement
        if movement == 1:  # Up
            self.ball_vel[1] -= self.BALL_ACCEL
        elif movement == 2:  # Down
            self.ball_vel[1] += self.BALL_ACCEL
        elif movement == 3:  # Left
            self.ball_vel[0] -= self.BALL_ACCEL
        elif movement == 4:  # Right
            self.ball_vel[0] += self.BALL_ACCEL

        # Shield Activation (on button press, not hold)
        if space_held and not self.prev_space_held:
            if not self.shield_active and self.score >= self.SHIELD_COST:
                self.score -= self.SHIELD_COST
                self.shield_active = True
                self.shield_timer = self.SHIELD_DURATION_STEPS
                # SFX: Shield Activate
                self._create_particle_burst(self.ball_pos, 30, self.COLOR_SHIELD, 2.0, 4.0)
        
        self.prev_space_held = space_held

    def _update_shield(self):
        if self.shield_active:
            self.shield_timer -= 1
            if self.shield_timer <= 0:
                self.shield_active = False
                # SFX: Shield Deactivate

    def _update_ball(self):
        # Apply damping and clamp velocity
        self.ball_vel *= self.BALL_DAMPING
        self.ball_vel = np.clip(self.ball_vel, -self.MAX_VEL, self.MAX_VEL)
        
        self.ball_pos += self.ball_vel
        
        reward = 0.0
        hit_wall = False

        # Wall collisions
        if self.ball_pos[0] < self.BALL_RADIUS:
            self.ball_pos[0] = self.BALL_RADIUS
            self.ball_vel[0] *= -1
            hit_wall = True
        elif self.ball_pos[0] > self.WIDTH - self.BALL_RADIUS:
            self.ball_pos[0] = self.WIDTH - self.BALL_RADIUS
            self.ball_vel[0] *= -1
            hit_wall = True
        
        if self.ball_pos[1] < self.BALL_RADIUS:
            self.ball_pos[1] = self.BALL_RADIUS
            self.ball_vel[1] *= -1
            hit_wall = True
        elif self.ball_pos[1] > self.HEIGHT - self.BALL_RADIUS:
            self.ball_pos[1] = self.HEIGHT - self.BALL_RADIUS
            self.ball_vel[1] *= -1
            hit_wall = True

        if hit_wall:
            if not self.shield_active:
                self.score -= 5
                reward -= 5.0
                self.wall_hit_flash = 5 # Flash for 5 frames
                # SFX: Wall Hit Penalty
            else:
                # SFX: Shield Wall Block
                self._create_particle_burst(self.ball_pos, 15, self.COLOR_SHIELD, 1.0, 3.0)
        
        return reward

    def _update_targets(self):
        reward = 0.0
        targets_to_remove = []
        for i, target in enumerate(self.targets):
            dist = np.linalg.norm(self.ball_pos - target['pos'])
            if dist < self.BALL_RADIUS + self.TARGET_RADIUS:
                self.score += 10
                reward += 10.0
                targets_to_remove.append(i)
                # SFX: Target Hit
                self._create_particle_burst(target['pos'], 40, self.COLOR_TARGET, 2.0, 5.0)

        if targets_to_remove:
            # Remove from back to front to avoid index errors
            for i in sorted(targets_to_remove, reverse=True):
                del self.targets[i]
            for _ in range(len(targets_to_remove)):
                self._spawn_target()
        
        return reward
    
    def _spawn_target(self):
        # Spawn away from walls and the ball
        padding = self.TARGET_RADIUS + 20
        pos = np.array([
            self.np_random.uniform(padding, self.WIDTH - padding),
            self.np_random.uniform(padding, self.HEIGHT - padding)
        ], dtype=np.float32)
        
        # Ensure it doesn't spawn too close to the ball
        while np.linalg.norm(pos - self.ball_pos) < self.TARGET_RADIUS + self.BALL_RADIUS + 50:
             pos = np.array([
                self.np_random.uniform(padding, self.WIDTH - padding),
                self.np_random.uniform(padding, self.HEIGHT - padding)
            ], dtype=np.float32)

        self.targets.append({'pos': pos, 'spawn_time': self.steps})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.1)

    def _create_particle_burst(self, pos, count, color, min_speed, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 30), # lifespan in frames
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _get_observation(self):
        # --- Render Game Elements ---
        self._render_background()
        self._render_particles()
        self._render_targets()
        self._render_ball()
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Draw border
        pygame.draw.rect(self.screen, self.COLOR_WALL_IDLE, (0, 0, self.WIDTH, self.HEIGHT), 2)

        # Flash effect on wall hit
        if self.wall_hit_flash > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = int(100 * (self.wall_hit_flash / 5.0))
            flash_surface.fill((255, 0, 0, alpha))
            self.screen.blit(flash_surface, (0, 0))
            self.wall_hit_flash -= 1

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color = (*p['color'], alpha)
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            size_int = int(p['size'])
            if size_int > 0:
                # Use a temporary surface for alpha blending
                particle_surf = pygame.Surface((size_int*2, size_int*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (size_int, size_int), size_int)
                self.screen.blit(particle_surf, (pos_int[0] - size_int, pos_int[1] - size_int))

    def _render_targets(self):
        for target in self.targets:
            pos_int = (int(target['pos'][0]), int(target['pos'][1]))
            # Pulsating effect based on spawn time
            pulse = (math.sin((self.steps - target['spawn_time']) * 0.2) + 1) / 2
            radius = int(self.TARGET_RADIUS * (0.9 + 0.1 * pulse))
            
            # Draw glow
            glow_radius = int(radius * 1.5)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*self.COLOR_TARGET, 30))
            self.screen.blit(glow_surf, (pos_int[0] - glow_radius, pos_int[1] - glow_radius))

            # Draw target
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, self.COLOR_TARGET)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, self.COLOR_TARGET)
    
    def _render_ball(self):
        pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        
        # Shield effect
        if self.shield_active:
            pulse = (math.sin(self.steps * 0.3) + 1) / 2
            shield_radius = int(self.BALL_RADIUS * (1.8 + 0.2 * pulse))
            alpha = 50 + int(100 * (self.shield_timer / self.SHIELD_DURATION_STEPS))
            
            # Draw multiple layers for a soft glow
            for i in range(3):
                r = int(shield_radius * (1 - i*0.15))
                a = int(alpha * (1 - i*0.2))
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], r, (*self.COLOR_SHIELD, a // 3))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], shield_radius, (*self.COLOR_SHIELD, alpha))

        # Ball glow
        glow_radius = self.BALL_RADIUS * 2
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*self.COLOR_BALL, 20))
        self.screen.blit(glow_surf, (pos_int[0] - glow_radius, pos_int[1] - glow_radius))
        
        # Ball core
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (10, 10))

        # Shield Status
        shield_status = "READY" if self.score >= self.SHIELD_COST and not self.shield_active else "UNAVAILABLE"
        shield_color = self.COLOR_SHIELD if shield_status == "READY" else self.COLOR_WALL_IDLE
        if self.shield_active:
            shield_status = "ACTIVE"
            shield_color = (255, 255, 255)
        
        shield_text = self.font_subtitle.render(f"SHIELD: {shield_status}", True, shield_color)
        self.screen.blit(shield_text, (10, 40))

        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_seconds = time_left / self.FPS
        timer_text = self.font_main.render(f"TIME: {time_seconds:.1f}", True, self.COLOR_TIMER)
        text_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(timer_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS,
            "ball_pos": self.ball_pos.tolist(),
            "ball_vel": self.ball_vel.tolist(),
            "shield_active": self.shield_active,
        }
        
    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # The validate_implementation method from the original code has been removed
    # as it is not part of the standard Gymnasium API and was for initial development.
    # The environment is expected to be validated by external test suites.

    env = GameEnv()
    
    # --- Manual Play ---
    obs, info = env.reset()
    
    # To run with a display, comment out the SDL_VIDEODRIVER line at the top
    # and uncomment the following lines.
    # os.environ.pop("SDL_VIDEODRIVER", None)
    # pygame.display.init()
    pygame.display.set_caption("Bouncing Ball Environment")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    # Map keys to actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    running = True
    while running:
        # Default action is no-op
        movement_action = 0
        space_action = 0
        shift_action = 0
        
        keys = pygame.key.get_pressed()
        
        for key, action_val in key_map.items():
            if keys[key]:
                movement_action = action_val
                break # Prioritize one movement key
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Wait 2 seconds
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        env.clock.tick(GameEnv.FPS)
        
    env.close()