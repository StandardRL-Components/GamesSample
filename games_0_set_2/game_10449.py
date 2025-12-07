import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a bouncing ball to hit targets.
    The goal is to score 500 points within 60 seconds by hitting targets, with
    bonuses for chaining hits together quickly.

    Visuals:
    - Clean, geometric style with glow effects and particle explosions.
    - High-contrast colors for clarity.

    Physics:
    - The ball is affected by gravity and player-applied forces.
    - It bounces elastically off the walls.

    Gameplay:
    - Hit green targets with the white ball.
    - Hitting targets in quick succession creates a score-multiplying chain.
    - The game ends when the score reaches 500 (win) or the timer runs out (loss).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a bouncing ball to hit targets and score points. "
        "Chain hits together for a score multiplier before the timer runs out."
    )
    user_guide = "Use the arrow keys (↑↓←→) to apply force to the ball and guide it into the targets."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    TIME_LIMIT_SECONDS = 60
    WIN_SCORE = 500

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_WALL = (80, 80, 90)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_TARGET = (0, 255, 120)
    COLOR_PARTICLE = (255, 200, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_CHAIN_TEXT = (255, 220, 50)

    # Physics
    GRAVITY = 0.2
    PLAYER_ACCEL = 0.6
    MAX_VEL = 8.0
    WALL_DAMPING = 1.0  # Perfect elasticity

    # Game Mechanics
    NUM_TARGETS = 3
    TARGET_RADIUS = 15
    PLAYER_RADIUS = 12
    TARGET_RESPAWN_TIME = 0.5 * FPS  # 0.5 seconds
    CHAIN_TIME_LIMIT = 2 * FPS      # 2 seconds to continue a chain

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- State Initialization ---
        self.ball_pos = None
        self.ball_vel = None
        self.targets = None
        self.particles = None
        self.score = None
        self.steps = None
        self.max_steps = self.TIME_LIMIT_SECONDS * self.FPS
        self.current_chain = None
        self.chain_timer = None
        self.game_over = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.ball_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.ball_vel = np.array([0.0, 0.0], dtype=np.float32)

        self.targets = []
        for _ in range(self.NUM_TARGETS):
            self._spawn_target()

        self.particles = []
        self.score = 0
        self.steps = 0
        self.current_chain = 0
        self.chain_timer = 0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]
        # space_held = action[1] == 1  # Unused in this game
        # shift_held = action[2] == 1 # Unused in this game

        # --- Update Game Logic ---
        self.steps += 1
        reward = 0

        self._update_player_movement(movement)
        self._update_physics()
        self._update_chain()
        self._update_particles()
        
        hit_reward = self._update_targets_and_collisions()
        reward += hit_reward

        # --- Check Termination Conditions ---
        terminated = self._check_termination()
        truncated = False # This game ends on a condition (score or time), not an artificial step limit
        if terminated and self.score >= self.WIN_SCORE:
            reward += 100  # Goal-oriented reward for winning

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player_movement(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1:  # Up
            self.ball_vel[1] -= self.PLAYER_ACCEL
        elif movement == 2:  # Down
            self.ball_vel[1] += self.PLAYER_ACCEL
        elif movement == 3:  # Left
            self.ball_vel[0] -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.ball_vel[0] += self.PLAYER_ACCEL
        
        # Clamp velocity to prevent excessive speeds
        vel_norm = np.linalg.norm(self.ball_vel)
        if vel_norm > self.MAX_VEL:
            self.ball_vel = self.ball_vel / vel_norm * self.MAX_VEL

    def _update_physics(self):
        # Apply gravity
        self.ball_vel[1] += self.GRAVITY

        # Update position
        self.ball_pos += self.ball_vel

        # Wall collisions
        if self.ball_pos[0] < self.PLAYER_RADIUS:
            self.ball_pos[0] = self.PLAYER_RADIUS
            self.ball_vel[0] *= -self.WALL_DAMPING
        elif self.ball_pos[0] > self.WIDTH - self.PLAYER_RADIUS:
            self.ball_pos[0] = self.WIDTH - self.PLAYER_RADIUS
            self.ball_vel[0] *= -self.WALL_DAMPING

        if self.ball_pos[1] < self.PLAYER_RADIUS:
            self.ball_pos[1] = self.PLAYER_RADIUS
            self.ball_vel[1] *= -self.WALL_DAMPING
        elif self.ball_pos[1] > self.HEIGHT - self.PLAYER_RADIUS:
            self.ball_pos[1] = self.HEIGHT - self.PLAYER_RADIUS
            self.ball_vel[1] *= -self.WALL_DAMPING

    def _update_targets_and_collisions(self):
        reward = 0
        for target in self.targets:
            if target['respawn_timer'] > 0:
                target['respawn_timer'] -= 1
                if target['respawn_timer'] == 0:
                    self._respawn_target(target)
            else:
                dist = np.linalg.norm(self.ball_pos - target['pos'])
                if dist < self.PLAYER_RADIUS + self.TARGET_RADIUS:
                    # --- HIT! ---
                    self.chain_timer = self.CHAIN_TIME_LIMIT
                    self.current_chain += 1

                    base_points = 10
                    chain_bonus_points = 5 * (self.current_chain - 1)
                    self.score += base_points + chain_bonus_points
                    
                    # Reward structure
                    reward += 0.1  # Continuous feedback
                    if self.current_chain == 1:
                        reward += 1.0  # Chain initiation
                    else:
                        reward += 2.0  # Chain continuation

                    target['respawn_timer'] = self.TARGET_RESPAWN_TIME
                    self._create_particle_explosion(target['pos'], self.COLOR_PARTICLE)
        return reward
    
    def _update_chain(self):
        if self.chain_timer > 0:
            self.chain_timer -= 1
            if self.chain_timer == 0:
                self.current_chain = 0

    def _update_particles(self):
        # Use a list comprehension for efficient filtering
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] -= 0.1  # Particles shrink over time

    def _check_termination(self):
        if self.score >= self.WIN_SCORE or self.steps >= self.max_steps:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "chain": self.current_chain,
            "time_left": (self.max_steps - self.steps) / self.FPS,
        }

    # --- Spawning and Creation ---
    def _spawn_target(self):
        # Find a valid position for a new target
        while True:
            pos = np.array([
                self.np_random.uniform(self.TARGET_RADIUS + 20, self.WIDTH - self.TARGET_RADIUS - 20),
                self.np_random.uniform(self.TARGET_RADIUS + 50, self.HEIGHT - self.TARGET_RADIUS - 20)
            ], dtype=np.float32)
            
            # Ensure it doesn't overlap with existing targets
            if not any(np.linalg.norm(pos - t['pos']) < 2 * self.TARGET_RADIUS for t in self.targets if t['respawn_timer'] == 0):
                break

        self.targets.append({'pos': pos, 'respawn_timer': 0})

    def _respawn_target(self, target_dict):
        # Find a new valid position
        while True:
            pos = np.array([
                self.np_random.uniform(self.TARGET_RADIUS + 20, self.WIDTH - self.TARGET_RADIUS - 20),
                self.np_random.uniform(self.TARGET_RADIUS + 50, self.HEIGHT - self.TARGET_RADIUS - 20)
            ], dtype=np.float32)
            
            # Check overlap with ball and other active targets
            valid = True
            if np.linalg.norm(pos - self.ball_pos) < self.TARGET_RADIUS + self.PLAYER_RADIUS + 20:
                valid = False
            if any(np.linalg.norm(pos - t['pos']) < 2 * self.TARGET_RADIUS for t in self.targets if t['respawn_timer'] == 0):
                valid = False
            
            if valid:
                target_dict['pos'] = pos
                break
    
    def _create_particle_explosion(self, pos, color):
        num_particles = 20
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': self.np_random.uniform(3, 7),
                'lifetime': self.np_random.integers(15, 30),
                'color': color
            })

    # --- Rendering ---
    def _render_game(self):
        # Render walls (for visual clarity)
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.HEIGHT), 2)
        
        # Render targets
        for target in self.targets:
            if target['respawn_timer'] == 0:
                self._draw_glow_circle(
                    self.screen,
                    target['pos'].astype(int),
                    self.TARGET_RADIUS,
                    self.COLOR_TARGET
                )

        # Render particles
        for p in self.particles:
            if p['radius'] > 0:
                self._draw_glow_circle(
                    self.screen,
                    p['pos'].astype(int),
                    int(p['radius']),
                    p['color']
                )

        # Render player
        self._draw_glow_circle(
            self.screen,
            self.ball_pos.astype(int),
            self.PLAYER_RADIUS,
            self.COLOR_PLAYER
        )

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score:04d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Timer
        time_left = max(0, (self.max_steps - self.steps) / self.FPS)
        time_text = self.font_large.render(f"{time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 5))

        # Chain multiplier
        if self.current_chain > 1:
            chain_str = f"x{self.current_chain} CHAIN!"
            
            # Animate size and position based on chain timer
            scale = 1.0 + 0.2 * math.sin((self.chain_timer / self.CHAIN_TIME_LIMIT) * math.pi)
            scaled_font = pygame.font.SysFont("Consolas", int(24 * scale), bold=True)
            chain_text = scaled_font.render(chain_str, True, self.COLOR_CHAIN_TEXT)
            
            text_pos = (self.WIDTH // 2 - chain_text.get_width() // 2, 10)
            self.screen.blit(chain_text, text_pos)

    def _draw_glow_circle(self, surface, pos, radius, color):
        """Renders a circle with a soft glow effect."""
        if radius <= 0: return
        
        # Outer, more transparent glow
        glow_radius = int(radius * 1.6)
        glow_alpha = 40
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*color, glow_alpha), (glow_radius, glow_radius), glow_radius)
        surface.blit(temp_surf, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Inner, less transparent glow
        glow_radius = int(radius * 1.2)
        glow_alpha = 60
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*color, glow_alpha), (glow_radius, glow_radius), glow_radius)
        surface.blit(temp_surf, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Solid core circle using anti-aliased drawing
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)

    def close(self):
        pygame.quit()

# --- Example Usage (for human play and testing) ---
if __name__ == '__main__':
    # Set a real video driver for the display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Play Loop ---
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Pygame window for display
    pygame.display.set_caption("Bouncing Ball Target Chain")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action[0] = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action[0] = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
        
        if keys[pygame.K_r]: # Reset game
             obs, info = env.reset()
             total_reward = 0
             done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # --- Rendering to the display window ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        if done:
            # Display game over message
            font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
            msg = "YOU WIN!" if info['score'] >= env.WIN_SCORE else "TIME UP!"
            text_surf = font_game_over.render(msg, True, (255, 255, 0))
            text_rect = text_surf.get_rect(center=(env.WIDTH/2, env.HEIGHT/2 - 20))
            screen.blit(text_surf, text_rect)
            
            font_restart = pygame.font.SysFont("Consolas", 20)
            restart_surf = font_restart.render("Press 'R' to restart", True, (200, 200, 200))
            restart_rect = restart_surf.get_rect(center=(env.WIDTH/2, env.HEIGHT/2 + 30))
            screen.blit(restart_surf, restart_rect)

        pygame.display.flip()
        env.clock.tick(env.FPS)

    env.close()