import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:09:58.159065
# Source Brief: brief_00300.md
# Brief Index: 300
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple class for managing particle effects."""
    def __init__(self, x, y, color, vx, vy, lifespan, radius):
        self.pos = [x, y]
        self.vel = [vx, vy]
        self.color = color
        self.lifespan = lifespan
        self.initial_lifespan = lifespan
        self.radius = radius

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifespan -= 1

    def draw(self, surface, camera_y):
        if self.lifespan > 0:
            # Fade out effect
            alpha = int(255 * (self.lifespan / self.initial_lifespan))
            # Shrink effect
            current_radius = int(self.radius * (self.lifespan / self.initial_lifespan))
            if current_radius > 0:
                # Using a temporary surface for alpha blending
                particle_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(
                    particle_surf,
                    (*self.color, alpha),
                    (current_radius, current_radius),
                    current_radius
                )
                surface.blit(particle_surf, (int(self.pos[0] - current_radius), int(self.pos[1] - camera_y - current_radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Descend through a dangerous tunnel, collecting green orbs to grow while avoiding red orbs and the walls to reach the goal."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to move left and right."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 900  # 30 seconds * 30 FPS

    # Colors
    COLOR_BG = (26, 26, 46)
    COLOR_WALL = (45, 45, 68)
    COLOR_PLAYER = (0, 170, 255)
    COLOR_PLAYER_GLOW = (100, 200, 255)
    COLOR_GREEN_ORB = (0, 255, 170)
    COLOR_RED_ORB = (255, 68, 102)
    COLOR_GOAL = (0, 204, 136)
    COLOR_TEXT = (255, 255, 255)

    # Game Mechanics
    TUNNEL_LENGTH = 5000
    TUNNEL_WIDTH_RATIO = 0.6
    INITIAL_RADIUS = 20
    MAX_RADIUS = 60
    PLAYER_ACCEL = 1.2
    PLAYER_FRICTION = 0.90
    PLAYER_GRAVITY = 1.5
    ORB_RADIUS = 12
    INITIAL_ORB_SPAWN_INTERVAL = 30 # steps
    DIFFICULTY_INCREASE_INTERVAL = 300 # steps (10 seconds)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_vel = [0, 0]
        self.player_radius = 0
        self.orbs = []
        self.particles = []
        self.camera_y = 0
        self.orb_spawn_timer = 0
        self.current_orb_spawn_interval = self.INITIAL_ORB_SPAWN_INTERVAL
        
        # Derived constants
        self.tunnel_pixel_width = self.WIDTH * self.TUNNEL_WIDTH_RATIO
        self.tunnel_left_x = (self.WIDTH - self.tunnel_pixel_width) / 2
        self.tunnel_right_x = self.WIDTH - self.tunnel_left_x
        self.goal_y = self.TUNNEL_LENGTH - self.HEIGHT / 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = [self.WIDTH / 2, self.INITIAL_RADIUS * 2]
        self.player_vel = [0, 0] # vx, vy
        self.player_radius = self.INITIAL_RADIUS
        
        self.orbs = []
        self.particles = []
        self.camera_y = 0
        
        self.orb_spawn_timer = 0
        self.current_orb_spawn_interval = self.INITIAL_ORB_SPAWN_INTERVAL
        
        # Pre-populate some orbs at the start
        for _ in range(20):
            self._spawn_orb(initial_spawn=True)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        
        # 1. Handle Input & Update Player
        self._handle_input(action)
        self._update_player()
        
        # 2. Update Game World
        self._update_orbs()
        self._update_particles()
        
        # 3. Handle Collisions and Game Logic
        reward += self._handle_collisions()

        # 4. Add continuous reward for survival
        reward += 0.01

        # 5. Check for termination
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        
        if movement == 3:  # Left
            self.player_vel[0] -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel[0] += self.PLAYER_ACCEL

    def _update_player(self):
        # Apply friction
        self.player_vel[0] *= self.PLAYER_FRICTION
        
        # Apply gravity (constant downward speed)
        self.player_vel[1] = self.PLAYER_GRAVITY
        
        # Update position
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]
        
        # Update camera to keep player vertically centered
        self.camera_y = self.player_pos[1] - self.HEIGHT / 2

    def _handle_collisions(self):
        reward = 0

        # Wall collisions
        impact_velocity = 0
        if self.player_pos[0] - self.player_radius < self.tunnel_left_x:
            self.player_pos[0] = self.tunnel_left_x + self.player_radius
            impact_velocity = abs(self.player_vel[0])
            self.player_vel[0] *= -0.5 # Bounce
        elif self.player_pos[0] + self.player_radius > self.tunnel_right_x:
            self.player_pos[0] = self.tunnel_right_x - self.player_radius
            impact_velocity = abs(self.player_vel[0])
            self.player_vel[0] *= -0.5 # Bounce

        if impact_velocity > 0.1:
            size_reduction = 0.01 * impact_velocity**2
            self.player_radius *= (1 - min(0.1, size_reduction))
            reward -= 0.5 * impact_velocity
            # sfx_wall_hit()
            self._create_particles(self.player_pos[0], self.player_pos[1], self.COLOR_WALL, 15)

        # Orb collisions
        for orb in self.orbs[:]:
            dist = math.hypot(self.player_pos[0] - orb['pos'][0], self.player_pos[1] - orb['pos'][1])
            if dist < self.player_radius + self.ORB_RADIUS:
                if orb['type'] == 'green':
                    self.player_radius *= 1.25
                    self.score += 1
                    reward += 1.0
                    # sfx_collect_green()
                    self._create_particles(orb['pos'][0], orb['pos'][1], self.COLOR_GREEN_ORB, 20)
                else: # red
                    self.player_radius *= 0.70
                    self.score -= 1
                    reward -= 1.0
                    # sfx_collect_red()
                    self._create_particles(orb['pos'][0], orb['pos'][1], self.COLOR_RED_ORB, 20)
                
                self.player_radius = min(self.player_radius, self.MAX_RADIUS)
                self.orbs.remove(orb)

        return reward

    def _update_orbs(self):
        # Spawn new orbs
        self.orb_spawn_timer += 1
        if self.orb_spawn_timer >= self.current_orb_spawn_interval:
            self.orb_spawn_timer = 0
            self._spawn_orb()

        # Increase difficulty
        if self.steps > 0 and self.steps % self.DIFFICULTY_INCREASE_INTERVAL == 0:
            self.current_orb_spawn_interval = max(5, self.current_orb_spawn_interval * 0.9)

        # Remove orbs that are far above the camera
        self.orbs = [o for o in self.orbs if o['pos'][1] > self.camera_y - self.ORB_RADIUS]

    def _spawn_orb(self, initial_spawn=False):
        x = self.np_random.uniform(self.tunnel_left_x + self.ORB_RADIUS, self.tunnel_right_x - self.ORB_RADIUS)
        
        if initial_spawn:
            y = self.np_random.uniform(self.HEIGHT, self.TUNNEL_LENGTH - self.HEIGHT)
        else:
            y = self.camera_y + self.HEIGHT + self.np_random.uniform(50, 200)

        orb_type = 'green' if self.np_random.random() > 0.4 else 'red'
        self.orbs.append({'pos': [x, y], 'type': orb_type})

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0:
                self.particles.remove(p)

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifespan = self.np_random.integers(15, 30)
            radius = self.np_random.uniform(1, 4)
            self.particles.append(Particle(x, y, color, vx, vy, lifespan, radius))

    def _check_termination(self):
        if self.player_radius <= 1:
            # sfx_lose_shrink()
            return True, -10.0
        if self.steps >= self.MAX_STEPS:
            # sfx_lose_timeout()
            return True, -10.0
        if self.player_pos[1] >= self.goal_y:
            # sfx_win()
            return True, 100.0
        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render tunnel walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.tunnel_left_x, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.tunnel_right_x, 0, self.tunnel_left_x, self.HEIGHT))

        # Render goal zone
        goal_rect_y = self.goal_y - self.camera_y
        if 0 < goal_rect_y < self.HEIGHT:
            pygame.draw.rect(self.screen, self.COLOR_GOAL, (self.tunnel_left_x, goal_rect_y, self.tunnel_pixel_width, self.HEIGHT))
        
        # Render orbs
        for orb in self.orbs:
            color = self.COLOR_GREEN_ORB if orb['type'] == 'green' else self.COLOR_RED_ORB
            pos = (int(orb['pos'][0]), int(orb['pos'][1] - self.camera_y))
            if -self.ORB_RADIUS < pos[1] < self.HEIGHT + self.ORB_RADIUS:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ORB_RADIUS, color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ORB_RADIUS, color)

        # Render particles
        for p in self.particles:
            p.draw(self.screen, self.camera_y)

        # Render player
        if self.player_radius > 0:
            player_screen_pos = (int(self.player_pos[0]), int(self.player_pos[1] - self.camera_y))
            radius = int(self.player_radius)
            
            glow_radius = int(radius * 1.5)
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (player_screen_pos[0] - glow_radius, player_screen_pos[1] - glow_radius))

            pygame.gfxdraw.aacircle(self.screen, player_screen_pos[0], player_screen_pos[1], radius, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_circle(self.screen, player_screen_pos[0], player_screen_pos[1], radius, self.COLOR_PLAYER)

    def _render_ui(self):
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = f"TIME: {time_left:.1f}"
        time_surf = self.font.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))
        
        score_text = f"SCORE: {self.score}"
        score_surf = self.font.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_surf, score_rect)
        
        progress = min(1.0, self.player_pos[1] / self.goal_y)
        bar_width = 200
        bar_height = 10
        bar_x = (self.WIDTH - bar_width) / 2
        bar_y = self.HEIGHT - 25
        pygame.draw.rect(self.screen, self.COLOR_WALL, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_GOAL, (bar_x, bar_y, bar_width * progress, bar_height), border_radius=5)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_radius": self.player_radius,
            "y_progress": self.player_pos[1] / self.goal_y
        }
    
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the game and play it manually.
    # It is not used by the evaluation system.
    
    # Un-comment the line below to run with display
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Tunnel Descent")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    action = np.array([0, 0, 0]) # [movement, unused, unused]

    # Remove the validation check from the main execution block
    # as it's not needed for playing.
    # try:
    #     env.validate_implementation()
    # except AttributeError:
    #     # In case the method doesn't exist in the provided code
    #     print("Skipping implementation validation.")


    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Action mapping: 0=No-op, 3=Left, 4=Right
        action[0] = 0 
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            done = True
            
        clock.tick(env.FPS)

    env.close()
    pygame.quit()