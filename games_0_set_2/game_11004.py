import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:17:35.501631
# Source Brief: brief_01004.md
# Brief Index: 1004
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a bouncing ball.
    The goal is to collect orbs to charge a powerful laser, then use the laser
    to clear rows of descending obstacles. The game is won by clearing 3 rows.
    The game is lost if the ball falls off the bottom of the screen.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Space button (0=released, 1=held) - used for laser
    - actions[2]: Shift button (0=released, 1=held) - unused

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Control a bouncing ball to collect orbs and charge a powerful laser. Aim and fire the laser to destroy rows of descending obstacles to win."
    user_guide = "Controls: Use ←→ to move the ball. Hold space to aim the laser at the ball's current height, and release to fire."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 10000

        # --- Colors ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_WALL = (80, 90, 110)
        self.COLOR_BALL = (0, 200, 255)
        self.COLOR_BALL_GLOW = (0, 100, 155)
        self.COLOR_ORB = (100, 255, 100)
        self.COLOR_OBSTACLE = (255, 80, 80)
        self.COLOR_OBSTACLE_GLOW = (150, 40, 40)
        self.COLOR_LASER_AIM = (255, 255, 0, 150)
        self.COLOR_LASER_BEAM = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_CHARGE_FULL = (255, 255, 0)
        self.COLOR_UI_CHARGE_EMPTY = (60, 60, 80)

        # --- Physics & Gameplay ---
        self.BALL_RADIUS = 12
        self.BALL_ACCEL = 1.2
        self.BALL_FRICTION = 0.98
        self.BALL_GRAVITY = 0.6
        self.BALL_BOUNCE_FACTOR = -0.85
        self.ORB_RADIUS = 8
        self.ORB_FALL_SPEED = 2.0
        self.ORBS_TO_CHARGE = 5
        self.OBSTACLE_SIZE = (40, 20)
        self.OBSTACLE_SPAWN_RATE_INITIAL = 0.5  # seconds
        self.OBSTACLE_SPAWN_RATE_INCREASE = 0.001 / self.FPS
        self.CHAIN_REACTION_CHANCE = 0.3

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- State Variables ---
        self.ball_pos = None
        self.ball_vel = None
        self.orbs = None
        self.obstacles = None
        self.particles = None
        self.ripples = None
        self.laser_beam = None

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.rows_cleared = 0

        self.orb_count = 0
        self.laser_charges = 0
        self.is_aiming_laser = False
        self.space_was_held = False
        self.laser_y_target = 0

        self.obstacle_spawn_timer = 0
        self.obstacle_spawn_rate = self.OBSTACLE_SPAWN_RATE_INITIAL
        
        self.screen_shake = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.ball_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.ball_vel = np.array([0.0, 0.0], dtype=np.float32)

        self.orbs = []
        self.obstacles = []
        self.particles = []
        self.ripples = []
        self.laser_beam = None

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.rows_cleared = 0

        self.orb_count = 0
        self.laser_charges = 1 # Start with one charge
        self.is_aiming_laser = False
        self.space_was_held = False

        self.obstacle_spawn_timer = 0
        self.obstacle_spawn_rate = self.OBSTACLE_SPAWN_RATE_INITIAL
        self.screen_shake = 0

        # Initial setup
        self._spawn_obstacle_row(self.HEIGHT - 50)
        for _ in range(5):
            self._spawn_orb()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)
        self.steps += 1
        reward = 0

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Game Logic ---
        reward += self._handle_input(movement, space_held)
        self._update_ball()
        reward += self._update_orbs()
        self._update_obstacles()
        self._update_effects()

        # --- Termination ---
        terminated = False
        truncated = False
        if self.ball_pos[1] > self.HEIGHT + self.BALL_RADIUS:
            # Sfx: Player lose
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.rows_cleared >= 3:
            # Sfx: Player win
            reward += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        self.score += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Update Sub-routines ---
    
    def _handle_input(self, movement, space_held):
        # Ball movement
        if movement == 3:  # Left
            self.ball_vel[0] -= self.BALL_ACCEL
        elif movement == 4:  # Right
            self.ball_vel[0] += self.BALL_ACCEL

        # Laser control
        # Start aiming on press
        if space_held and not self.space_was_held and self.laser_charges > 0:
            self.is_aiming_laser = True
            # Sfx: Laser aim start
        
        # Stop aiming if charges are used up elsewhere
        if self.laser_charges == 0:
            self.is_aiming_laser = False

        # Fire laser on release
        if not space_held and self.space_was_held and self.is_aiming_laser:
            self.is_aiming_laser = False
            self.laser_charges -= 1
            # Sfx: Laser fire
            self.screen_shake = 15
            return self._fire_laser(self.laser_y_target)
        
        self.space_was_held = space_held
        return 0

    def _update_ball(self):
        # Apply physics
        self.ball_vel[1] += self.BALL_GRAVITY
        self.ball_vel *= self.BALL_FRICTION
        self.ball_pos += self.ball_vel

        # Wall collisions
        if self.ball_pos[0] < self.BALL_RADIUS:
            self.ball_pos[0] = self.BALL_RADIUS
            self.ball_vel[0] *= self.BALL_BOUNCE_FACTOR
            # Sfx: Ball bounce
        elif self.ball_pos[0] > self.WIDTH - self.BALL_RADIUS:
            self.ball_pos[0] = self.WIDTH - self.BALL_RADIUS
            self.ball_vel[0] *= self.BALL_BOUNCE_FACTOR
            # Sfx: Ball bounce
        if self.ball_pos[1] < self.BALL_RADIUS:
            self.ball_pos[1] = self.BALL_RADIUS
            self.ball_vel[1] *= self.BALL_BOUNCE_FACTOR
            # Sfx: Ball bounce

        # Obstacle collisions
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
        for obs_rect in self.obstacles:
            if ball_rect.colliderect(obs_rect):
                # Simple vertical bounce
                self.ball_vel[1] *= self.BALL_BOUNCE_FACTOR
                self.ball_pos[1] += self.ball_vel[1] # Push out
                # Sfx: Ball bounce on obstacle
                break

    def _update_orbs(self):
        reward = 0
        orbs_to_remove = []
        for orb in self.orbs:
            orb['pos'][1] += self.ORB_FALL_SPEED
            if orb['pos'][1] > self.HEIGHT + self.ORB_RADIUS:
                orbs_to_remove.append(orb)

            # Collision with ball
            dist = np.linalg.norm(self.ball_pos - orb['pos'])
            if dist < self.BALL_RADIUS + self.ORB_RADIUS:
                orbs_to_remove.append(orb)
                reward += 0.1
                self.orb_count += 1
                self._create_ripple(orb['pos'], self.COLOR_ORB)
                # Sfx: Orb collect
                if self.orb_count >= self.ORBS_TO_CHARGE:
                    self.orb_count = 0
                    if self.laser_charges < 3:
                        self.laser_charges += 1
                        # Sfx: Laser charge complete
        
        self.orbs = [o for o in self.orbs if o not in orbs_to_remove]

        # Respawn orbs
        if len(self.orbs) < 10 and self.np_random.random() < 0.05:
            self._spawn_orb()
            
        return reward

    def _update_obstacles(self):
        self.obstacle_spawn_timer += 1 / self.FPS
        self.obstacle_spawn_rate -= self.OBSTACLE_SPAWN_RATE_INCREASE
        self.obstacle_spawn_rate = max(0.2, self.obstacle_spawn_rate)

        if self.obstacle_spawn_timer > self.obstacle_spawn_rate:
            self.obstacle_spawn_timer = 0
            self._spawn_obstacle_row(20)

    def _update_effects(self):
        # Particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        
        # Ripples
        self.ripples = [r for r in self.ripples if r['life'] > 0]
        for r in self.ripples:
            r['radius'] += r['speed']
            r['life'] -= 1

        # Laser beam
        if self.laser_beam and self.laser_beam['life'] > 0:
            self.laser_beam['life'] -= 1
        else:
            self.laser_beam = None

        # Screen shake
        if self.screen_shake > 0:
            self.screen_shake -= 1

    # --- Spawning and Actions ---

    def _spawn_orb(self):
        pos = np.array([self.np_random.uniform(self.ORB_RADIUS, self.WIDTH - self.ORB_RADIUS), -self.ORB_RADIUS], dtype=np.float32)
        self.orbs.append({'pos': pos})

    def _spawn_obstacle_row(self, y_pos):
        w, h = self.OBSTACLE_SIZE
        gap_index = self.np_random.integers(0, self.WIDTH // w)
        for i in range(self.WIDTH // w):
            if i != gap_index:
                rect = pygame.Rect(i * w, y_pos, w, h)
                self.obstacles.append(rect)

    def _fire_laser(self, y_pos):
        reward = 0
        destroyed_this_shot = []
        
        # Find all obstacles in the laser's path
        laser_line = pygame.Rect(0, y_pos - 2, self.WIDTH, 4)
        obstacles_in_row = [obs for obs in self.obstacles if laser_line.colliderect(obs)]

        if not obstacles_in_row:
             self.laser_beam = {'y': y_pos, 'life': 10, 'is_row_clear': False}
             return 0

        # Initial destruction
        for obs in obstacles_in_row:
            if obs not in destroyed_this_shot:
                destroyed_this_shot.append(obs)
                reward += 1.0 # Laser destruction reward
                self._create_particles(obs.center, self.COLOR_OBSTACLE, 20)

        # Chain reaction
        q = deque(obstacles_in_row)
        checked_for_chain = set(tuple(o.topleft) for o in q)

        while q:
            current_obs = q.popleft()
            # Check neighbors (up, down, left, right)
            for dx, dy in [(0, -self.OBSTACLE_SIZE[1]), (0, self.OBSTACLE_SIZE[1]), (-self.OBSTACLE_SIZE[0], 0), (self.OBSTACLE_SIZE[0], 0)]:
                neighbor_pos = (current_obs.left + dx, current_obs.top + dy)
                if neighbor_pos in checked_for_chain:
                    continue
                checked_for_chain.add(neighbor_pos)

                for other_obs in self.obstacles:
                    if other_obs.topleft == neighbor_pos and other_obs not in destroyed_this_shot:
                        if self.np_random.random() < self.CHAIN_REACTION_CHANCE:
                            destroyed_this_shot.append(other_obs)
                            reward += 0.2 # Chain reaction reward
                            self._create_particles(other_obs.center, self.COLOR_OBSTACLE, 10)
                            q.append(other_obs)
                            # Sfx: Chain explosion
                        break
        
        # Remove all destroyed obstacles
        self.obstacles = [obs for obs in self.obstacles if obs not in destroyed_this_shot]

        # Row clear bonus
        self.rows_cleared += 1
        reward += 10.0
        # Sfx: Row clear
        
        self.laser_beam = {'y': y_pos, 'life': 10, 'is_row_clear': True}
        return reward

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
            self.particles.append({'pos': np.copy(pos), 'vel': vel, 'life': self.np_random.integers(15, 30), 'color': color})

    def _create_ripple(self, pos, color):
        self.ripples.append({'pos': pos, 'radius': 5, 'life': 20, 'speed': 2, 'color': color})

    # --- Rendering ---

    def _get_observation(self):
        render_surface = self.screen
        if self.screen_shake > 0:
            offset_x = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
            offset_y = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
            render_offset = (offset_x, offset_y)
        else:
            render_offset = (0, 0)
        
        # Create a temporary surface to blit onto for the shake effect
        temp_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        
        # Draw all elements onto the temp surface
        self._render_background(temp_surface)
        self._render_ripples(temp_surface)
        self._render_obstacles(temp_surface)
        self._render_orbs(temp_surface)
        self._render_ball(temp_surface)
        self._render_particles(temp_surface)
        self._render_laser_effects(temp_surface)
        self._render_ui(temp_surface)
        if self.game_over:
            self._render_game_over(temp_surface)

        # Blit the temp surface to the main screen with the shake offset
        self.screen.fill(self.COLOR_BG)
        self.screen.blit(temp_surface, render_offset)
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, surface):
        surface.fill(self.COLOR_BG)
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(surface, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(surface, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)

    def _render_ripples(self, surface):
        for r in self.ripples:
            alpha = int(255 * (r['life'] / 20))
            pygame.gfxdraw.aacircle(surface, int(r['pos'][0]), int(r['pos'][1]), int(r['radius']), (*r['color'], alpha))

    def _render_obstacles(self, surface):
        for obs in self.obstacles:
            pygame.draw.rect(surface, self.COLOR_OBSTACLE_GLOW, obs.inflate(6, 6), border_radius=4)
            pygame.draw.rect(surface, self.COLOR_OBSTACLE, obs, border_radius=4)

    def _render_orbs(self, surface):
        for orb in self.orbs:
            pos = (int(orb['pos'][0]), int(orb['pos'][1]))
            pygame.gfxdraw.filled_circle(surface, *pos, self.ORB_RADIUS, self.COLOR_ORB)
            pygame.gfxdraw.aacircle(surface, *pos, self.ORB_RADIUS, self.COLOR_ORB)

    def _render_ball(self, surface):
        pos = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        # Glow effect
        for i in range(5):
            alpha = 100 - i * 20
            pygame.gfxdraw.filled_circle(surface, *pos, self.BALL_RADIUS + i * 2, (*self.COLOR_BALL_GLOW, alpha))
        # Ball
        pygame.gfxdraw.filled_circle(surface, *pos, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(surface, *pos, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_particles(self, surface):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.circle(surface, color, pos, 2)

    def _render_laser_effects(self, surface):
        # Aiming reticle
        if self.is_aiming_laser:
            self.laser_y_target = self.ball_pos[1]
            y = int(self.laser_y_target)
            aim_surface = pygame.Surface((self.WIDTH, 3), pygame.SRCALPHA)
            aim_surface.fill(self.COLOR_LASER_AIM)
            surface.blit(aim_surface, (0, y - 1))

        # Fired laser beam
        if self.laser_beam:
            y = int(self.laser_beam['y'])
            life_ratio = self.laser_beam['life'] / 10.0
            width = int(life_ratio * 20) if self.laser_beam['is_row_clear'] else int(life_ratio * 4)
            alpha = int(life_ratio * 255)
            
            laser_surf = pygame.Surface((self.WIDTH, width), pygame.SRCALPHA)
            laser_surf.fill((*self.COLOR_LASER_BEAM, alpha))
            surface.blit(laser_surf, (0, y - width // 2))

    def _render_ui(self, surface):
        # Orb progress bar
        bar_width = 150
        bar_height = 20
        progress = self.orb_count / self.ORBS_TO_CHARGE
        pygame.draw.rect(surface, self.COLOR_UI_CHARGE_EMPTY, (10, 10, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(surface, self.COLOR_ORB, (10, 10, int(bar_width * progress), bar_height), border_radius=4)
        orb_text = self.font_ui.render(f"ORBS", True, self.COLOR_UI_TEXT)
        surface.blit(orb_text, (15, 8))

        # Laser charges
        charge_text = self.font_ui.render(f"LASER", True, self.COLOR_UI_TEXT)
        surface.blit(charge_text, (self.WIDTH - 150, 8))
        for i in range(3):
            color = self.COLOR_UI_CHARGE_FULL if i < self.laser_charges else self.COLOR_UI_CHARGE_EMPTY
            pygame.draw.rect(surface, color, (self.WIDTH - 80 + i * 25, 10, 20, 20), border_radius=4)
        
        # Rows cleared
        rows_text = self.font_ui.render(f"ROWS: {self.rows_cleared}/3", True, self.COLOR_UI_TEXT)
        text_rect = rows_text.get_rect(center=(self.WIDTH/2, 20))
        surface.blit(rows_text, text_rect)

    def _render_game_over(self, surface):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        surface.blit(overlay, (0, 0))
        
        msg = "VICTORY!" if self.rows_cleared >= 3 else "GAME OVER"
        text = self.font_msg.render(msg, True, self.COLOR_UI_TEXT)
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        surface.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "laser_charges": self.laser_charges,
            "rows_cleared": self.rows_cleared,
        }
        
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires pygame to be installed and will open a window.
    # The main environment does not require a display.
    os.environ.unsetenv("SDL_VIDEODRIVER")
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Bouncing Laser Orb")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space_held = 1

        action = [movement, space_held, 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # The observation is already a rendered image, so we just need to display it
        # Pygame uses (W, H) but the obs is (H, W, C), so we need to transpose
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Info: {info}")
            # Wait for 'R' to reset
            wait_for_reset = True
            while wait_for_reset:
                 for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False

        clock.tick(env.FPS)
        
    env.close()