import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# Generated: 2025-08-26T12:11:10.412261
# Source Brief: brief_00936.md
# Brief Index: 936
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player defends a fortress from waves of projectiles.
    The player places and modifies portals to redirect enemy fire.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Defend your fortress from waves of projectiles by placing and modifying portals to redirect enemy fire."
    user_guide = "Use arrow keys (↑↓←→) to move the selector. Press Shift to place a portal and Space to switch its state."
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000
    TOTAL_WAVES = 20
    MAX_PORTALS = 4

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_FORTRESS = (100, 100, 120)
    COLOR_FORTRESS_WALL = (150, 150, 170)
    COLOR_SELECTOR = (255, 255, 0)

    # Portal States
    PORTAL_STATE_GRAVITY = 0
    PORTAL_STATE_REFLECT = 1
    COLOR_GRAVITY = (50, 100, 255)
    COLOR_REFLECT = (255, 150, 50)

    # Projectile Types
    PROJ_GREEN = {'color': (80, 255, 80), 'damage': 5, 'radius': 5}
    PROJ_BLUE = {'color': (80, 150, 255), 'damage': 10, 'radius': 7}
    PROJ_RED = {'color': (255, 80, 80), 'damage': 20, 'radius': 9}
    
    # Grid for portal placement
    GRID_COLS, GRID_ROWS = 10, 5
    GRID_CELL_W = WIDTH // GRID_COLS
    GRID_CELL_H = 60
    GRID_Y_OFFSET = 50

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Game state variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.fortress_health = None
        self.max_fortress_health = None
        self.wave_number = None
        self.projectiles = None
        self.portals = None
        self.particles = None
        self.selector_pos = None
        self.last_space_held = None
        self.last_shift_held = None
        self.wave_in_progress = None
        self.wave_transition_timer = None
        self.move_cooldown = None
        self.reward_this_step = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.max_fortress_health = 100
        self.fortress_health = self.max_fortress_health
        
        self.wave_number = 1
        self.wave_in_progress = False
        self.wave_transition_timer = self.FPS * 3 # 3 second delay for first wave
        
        self.projectiles = []
        self.portals = []
        self.particles = []
        
        self.selector_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.last_space_held = False
        self.last_shift_held = False
        self.move_cooldown = 0
        
        self.reward_this_step = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        terminated = False
        truncated = False

        if self.game_over:
            # The episode is over, but we need to return a valid observation.
            # The last rendered frame is fine.
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game Logic ---
        self._update_wave_spawning()
        self._update_projectiles()
        self._update_particles()
        
        # --- Update State ---
        self.steps += 1
        
        # --- Check Termination Conditions ---
        if self.fortress_health <= 0:
            self.reward_this_step -= 100 # Large penalty for losing
            terminated = True
            self.game_over = True
            # sfx: fortress_destroyed_sound
        elif self.wave_number > self.TOTAL_WAVES:
            self.reward_this_step += 100 # Large reward for winning
            terminated = True
            self.game_over = True
            # sfx: victory_fanfare
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        # Clamp score to prevent runaway values from bugs
        self.score = max(0, self.score)

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Selector Movement with Cooldown ---
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        
        if self.move_cooldown == 0 and movement != 0:
            self.move_cooldown = 5 # frames
            if movement == 1: self.selector_pos[1] -= 1 # Up
            elif movement == 2: self.selector_pos[1] += 1 # Down
            elif movement == 3: self.selector_pos[0] -= 1 # Left
            elif movement == 4: self.selector_pos[0] += 1 # Right
            
            # Wrap selector around grid
            self.selector_pos[0] %= self.GRID_COLS
            self.selector_pos[1] %= self.GRID_ROWS
            # sfx: selector_move_tick

        # --- Portal State Switch (on key press) ---
        if space_held and not self.last_space_held:
            for portal in self.portals:
                if portal['grid_pos'] == self.selector_pos:
                    portal['state'] = 1 - portal['state'] # Toggle 0 and 1
                    # sfx: portal_switch_sound
                    break
        
        # --- Portal Placement (on key press) ---
        if shift_held and not self.last_shift_held:
            is_occupied = any(p['grid_pos'] == self.selector_pos for p in self.portals)
            if not is_occupied and len(self.portals) < self.MAX_PORTALS:
                new_portal = {
                    'grid_pos': list(self.selector_pos),
                    'state': self.PORTAL_STATE_GRAVITY,
                    'anim_timer': random.uniform(0, 2 * math.pi)
                }
                self.portals.append(new_portal)
                # sfx: portal_place_sound
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_wave_spawning(self):
        if not self.wave_in_progress:
            if self.wave_transition_timer > 0:
                self.wave_transition_timer -= 1
            else:
                self._spawn_wave()
                self.wave_in_progress = True
                # sfx: wave_start_alert
        elif not self.projectiles: # Wave is cleared
            self.wave_in_progress = False
            self.wave_transition_timer = self.FPS * 5 # 5 seconds between waves
            self.wave_number += 1
            if self.wave_number <= self.TOTAL_WAVES:
                self.reward_this_step += 1.0 # Reward for surviving a wave
                self.score += 100

    def _spawn_wave(self):
        num_projectiles = 5 + (self.wave_number - 1) // 2
        base_speed = 2.0 + (self.wave_number - 1) * 0.1
        
        for _ in range(num_projectiles):
            x_start = random.uniform(20, self.WIDTH - 20)
            y_start = -20.0
            
            angle = math.radians(random.uniform(70, 110))
            speed = random.uniform(base_speed * 0.8, base_speed * 1.2)
            
            # Choose projectile type based on wave
            rand_val = random.random()
            if self.wave_number > 10 and rand_val > 0.6:
                proj_type = self.PROJ_RED
            elif self.wave_number > 5 and rand_val > 0.4:
                proj_type = self.PROJ_BLUE
            else:
                proj_type = self.PROJ_GREEN
            
            self.projectiles.append({
                'pos': pygame.Vector2(x_start, y_start),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'type': proj_type,
                'last_portal_hit': None,
                'trail': []
            })

    def _update_projectiles(self):
        fortress_rect = pygame.Rect(0, self.HEIGHT - 20, self.WIDTH, 20)
        
        for p in self.projectiles[:]:
            p['trail'].append(pygame.Vector2(p['pos']))
            if len(p['trail']) > 15:
                p['trail'].pop(0)

            p['pos'] += p['vel']
            
            # Portal interaction
            collided_portal = self._check_portal_collision(p)
            if collided_portal and collided_portal != p['last_portal_hit']:
                p['last_portal_hit'] = collided_portal
                self.reward_this_step += 0.1
                self.score += 10
                # sfx: portal_redirect_sound
                if collided_portal['state'] == self.PORTAL_STATE_REFLECT:
                    p['vel'].reflect_ip(pygame.Vector2(0, 1)) # Simple vertical reflection
                    p['vel'] *= 1.1 # Speed up on reflection
                elif collided_portal['state'] == self.PORTAL_STATE_GRAVITY:
                    p['vel'].y += 0.5 # Apply gravity pull
                    p['vel'].y = min(p['vel'].y, 8) # Terminal velocity

            # Fortress collision
            if fortress_rect.collidepoint(p['pos']):
                self.fortress_health -= p['type']['damage']
                self._create_impact_particles(p['pos'], p['type']['color'])
                self.projectiles.remove(p)
                # sfx: fortress_hit_sound
                continue
            
            # Out of bounds
            if not ( -50 < p['pos'].x < self.WIDTH + 50 and -50 < p['pos'].y < self.HEIGHT + 50):
                self.projectiles.remove(p)

    def _check_portal_collision(self, projectile):
        proj_pos = projectile['pos']
        for portal in self.portals:
            px, py = self._grid_to_pixel(portal['grid_pos'])
            portal_rect = pygame.Rect(px, py, self.GRID_CELL_W, self.GRID_CELL_H)
            if portal_rect.collidepoint(proj_pos):
                return portal
        return None

    def _update_particles(self):
        for particle in self.particles[:]:
            particle['pos'] += particle['vel']
            particle['lifespan'] -= 1
            if particle['lifespan'] <= 0:
                self.particles.remove(particle)

    def _create_impact_particles(self, pos, color):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'lifespan': random.randint(15, 30),
                'color': color,
                'radius': random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_portals()
        self._render_fortress()
        self._render_projectiles()
        self._render_particles()
        self._render_portal_selector()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_fortress(self):
        fortress_rect = pygame.Rect(0, self.HEIGHT - 20, self.WIDTH, 20)
        wall_rect = pygame.Rect(0, self.HEIGHT - 20, self.WIDTH, 5)
        pygame.draw.rect(self.screen, self.COLOR_FORTRESS, fortress_rect)
        pygame.draw.rect(self.screen, self.COLOR_FORTRESS_WALL, wall_rect)

    def _render_portals(self):
        for portal in self.portals:
            px, py = self._grid_to_pixel(portal['grid_pos'])
            rect = pygame.Rect(px, py, self.GRID_CELL_W, self.GRID_CELL_H)
            
            # Animate portal
            portal['anim_timer'] += 0.1
            
            # Determine color and draw effect
            is_gravity = portal['state'] == self.PORTAL_STATE_GRAVITY
            color = self.COLOR_GRAVITY if is_gravity else self.COLOR_REFLECT
            
            # Swirling effect
            num_points = 5
            for i in range(num_points):
                angle = portal['anim_timer'] + (2 * math.pi * i / num_points)
                radius_factor = 0.4 + 0.1 * math.sin(portal['anim_timer'] * 2 + i)
                radius = min(self.GRID_CELL_W, self.GRID_CELL_H) * radius_factor
                
                particle_x = rect.centerx + radius * math.cos(angle)
                particle_y = rect.centery + radius * math.sin(angle if is_gravity else -angle)
                
                size = 3 + 2 * math.sin(portal['anim_timer'] + i)
                pygame.draw.circle(self.screen, color, (int(particle_x), int(particle_y)), int(max(0, size)))

            pygame.draw.rect(self.screen, (200, 200, 220), rect, 2, border_radius=5)

    def _render_projectiles(self):
        for p in self.projectiles:
            # Draw trail
            if len(p['trail']) > 1:
                for i, pos in enumerate(p['trail']):
                    alpha = int(255 * (i / len(p['trail'])))
                    color = (*p['type']['color'], alpha)
                    temp_surf = pygame.Surface((p['type']['radius']*2, p['type']['radius']*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, color, (p['type']['radius'], p['type']['radius']), int(p['type']['radius'] * (i / len(p['trail']))))
                    self.screen.blit(temp_surf, (int(pos.x - p['type']['radius']), int(pos.y - p['type']['radius'])))

            # Draw main projectile with glow
            pos_int = (int(p['pos'].x), int(p['pos'].y))
            radius = p['type']['radius']
            color = p['type']['color']
            
            for i in range(4, 0, -1):
                glow_color = (*color, 50 - i*10)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius + i, glow_color)
            
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, color)

    def _render_particles(self):
        for particle in self.particles:
            alpha = int(255 * (particle['lifespan'] / 30.0))
            color = (*particle['color'], alpha)
            pos_int = (int(particle['pos'].x), int(particle['pos'].y))
            radius = int(particle['radius'] * (particle['lifespan'] / 30.0))
            if radius > 0:
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (pos_int[0] - radius, pos_int[1] - radius))

    def _render_portal_selector(self):
        px, py = self._grid_to_pixel(self.selector_pos)
        rect = pygame.Rect(px, py, self.GRID_CELL_W, self.GRID_CELL_H)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, rect, 2, border_radius=5)

    def _render_ui(self):
        # Wave Text
        wave_text = f"WAVE {self.wave_number}/{self.TOTAL_WAVES}"
        text_surf = self.font_small.render(wave_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        # Score Text
        score_text = f"SCORE: {self.score}"
        text_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))

        # Health Bar
        health_ratio = max(0, self.fortress_health / self.max_fortress_health)
        bar_width = self.WIDTH - 20
        health_bar_width = int(bar_width * health_ratio)
        health_color = (int(255 * (1 - health_ratio)), int(255 * health_ratio), 50)
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 35, bar_width, 10))
        if health_bar_width > 0:
            pygame.draw.rect(self.screen, health_color, (10, 35, health_bar_width, 10))
            
        # Wave transition text
        if not self.wave_in_progress and self.wave_number <= self.TOTAL_WAVES:
            transition_text = f"WAVE {self.wave_number} INCOMING"
            text_surf = self.font_large.render(transition_text, True, self.COLOR_UI_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)
            
        # Game Over Text
        if self.game_over:
            outcome_text = "VICTORY" if self.wave_number > self.TOTAL_WAVES else "FORTRESS DESTROYED"
            text_surf = self.font_large.render(outcome_text, True, (255, 50, 50))
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "health": self.fortress_health,
            "portals": len(self.portals)
        }

    def _grid_to_pixel(self, grid_pos):
        x = grid_pos[0] * self.GRID_CELL_W
        y = grid_pos[1] * self.GRID_CELL_H + self.GRID_Y_OFFSET
        return x, y

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game with a human interface
    # It is not used by the evaluation system but is helpful for debugging
    
    # Un-set the dummy driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Portal Defense")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Human input mapping
        action = [0, 0, 0] # Default to no-op
        movement = 0 # None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_ESCAPE:
                    running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Info: {info}")
            # Keep the window open for a moment to see the result
            pygame.time.wait(2000)
            running = False 

        clock.tick(GameEnv.FPS)

    env.close()