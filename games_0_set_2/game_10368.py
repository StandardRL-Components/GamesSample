import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:16:28.276991
# Source Brief: brief_00368.md
# Brief Index: 368
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend the core by controlling three laser cannons. Intercept incoming projectiles before they pass through the defense perimeter."
    )
    user_guide = (
        "Controls: Use ←→ arrows to aim the selected cannon and ↑↓ arrows to adjust its power. "
        "Press space to fire the laser and shift to switch between cannons."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    FPS = 30
    MAX_STEPS = 1500

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_GRID = (20, 30, 50)
    COLOR_CANNON_INACTIVE = (50, 80, 120)
    COLOR_CANNON_ACTIVE = (200, 220, 255)
    COLOR_LASER = (0, 150, 255)
    COLOR_LASER_GLOW = (0, 50, 100)
    COLOR_PROJECTILE = (255, 50, 50)
    COLOR_PROJECTILE_GLOW = (100, 20, 20)
    COLOR_EXPLOSION = (255, 200, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BAR = (40, 60, 90)
    COLOR_UI_POWER = (0, 200, 255)
    
    # Game Parameters
    TOTAL_PROJECTILES = 15
    CANNON_RADIUS = 60
    CANNON_SIZE = 12
    LASER_ROTATION_SPEED = 2.0  # degrees per frame
    LASER_POWER_STEP = 0.1
    PROJECTILE_RADIUS = 7
    CHAIN_REACTION_RADIUS = 80
    CHAIN_REACTION_POWER_THRESHOLD = 1.5 # Combined power of > 150%

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # These state variables are initialized here but defined in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cannons = []
        self.projectiles = []
        self.laser_beams = []
        self.particles = []
        self.selected_cannon_idx = 0
        self.projectiles_intercepted = 0
        self.projectiles_passed = 0
        self.base_projectile_speed = 0.0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.reset()
        
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.projectiles_intercepted = 0
        self.projectiles_passed = 0
        self.base_projectile_speed = 1.0

        self.selected_cannon_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self._setup_cannons()
        self._spawn_projectiles()
        
        self.laser_beams = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_action, shift_action = action
        space_held = space_action == 1
        shift_held = shift_action == 1

        reward = 0
        self.steps += 1

        # --- Handle Input ---
        self._handle_input(movement, space_held, shift_held)

        # --- Update Game State ---
        self._update_projectiles()
        self._update_laser_beams()
        self._update_particles()

        # --- Collision Detection and Resolution ---
        hit_events, chain_events = self._process_collisions()
        
        # --- Calculate Reward ---
        reward += len(hit_events) * 1.0 # +1 for each projectile destroyed
        reward += len(chain_events) * 5.0 # +5 for each chain reaction
        
        # Update score and projectile counts based on events
        destroyed_this_step = set()
        for hit in hit_events:
            destroyed_this_step.add(hit['proj_idx'])
        for chain in chain_events:
            for proj_idx in chain['chained_indices']:
                destroyed_this_step.add(proj_idx)
        
        if destroyed_this_step:
            # Sound: Explosion
            for proj_idx in destroyed_this_step:
                if self.projectiles[proj_idx]['active']:
                    self.projectiles[proj_idx]['active'] = False
                    self.projectiles_intercepted += 1
                    self.score += 10 # Base score for any destruction
                    self._create_explosion(self.projectiles[proj_idx]['pos'])
            self.score += len(chain_events) * 50 # Bonus score for chains
        
        # Update projectile speed based on total interceptions
        speed_increase = (self.projectiles_intercepted // 3) * 0.02
        current_projectile_speed = self.base_projectile_speed + speed_increase
        for p in self.projectiles:
            if p['active'] and p['vel'].length() > 0:
                p['vel'] = p['vel'].normalize() * current_projectile_speed

        # --- Check Termination ---
        terminated = False
        active_projectiles = self.projectiles_intercepted + self.projectiles_passed
        if active_projectiles >= self.TOTAL_PROJECTILES:
            terminated = True
            if self.projectiles_intercepted == self.TOTAL_PROJECTILES:
                reward += 100 # Win bonus
                self.score += 1000
            else:
                reward -= 100 # Loss penalty
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            # Penalty for not finishing in time
            reward -= (self.TOTAL_PROJECTILES - self.projectiles_intercepted) * 5.0

        self.game_over = terminated
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    def _setup_cannons(self):
        self.cannons = []
        for i in range(3):
            angle = -90 + 120 * i
            x = self.CENTER[0] + self.CANNON_RADIUS * math.cos(math.radians(angle))
            y = self.CENTER[1] + self.CANNON_RADIUS * math.sin(math.radians(angle))
            self.cannons.append({
                'pos': pygame.Vector2(x, y),
                'angle': 0.0,
                'power': 0.5, # Start at 50%
            })
    
    def _spawn_projectiles(self):
        self.projectiles = []
        for _ in range(self.TOTAL_PROJECTILES):
            # Spawn on edge of screen
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -self.PROJECTILE_RADIUS)
            elif edge == 1: # Bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.PROJECTILE_RADIUS)
            elif edge == 2: # Left
                pos = pygame.Vector2(-self.PROJECTILE_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            else: # Right
                pos = pygame.Vector2(self.SCREEN_WIDTH + self.PROJECTILE_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT))

            # Aim towards the general center area with some randomness
            target_pos = pygame.Vector2(
                self.CENTER[0] + self.np_random.uniform(-100, 100),
                self.CENTER[1] + self.np_random.uniform(-100, 100)
            )
            vel = (target_pos - pos).normalize() * self.base_projectile_speed
            
            self.projectiles.append({
                'pos': pos,
                'vel': vel,
                'radius': self.PROJECTILE_RADIUS,
                'active': True
            })

    def _handle_input(self, movement, space_held, shift_held):
        cannon = self.cannons[self.selected_cannon_idx]

        # Movement: Rotate and change power
        if movement == 1: # Up
            cannon['power'] = min(1.0, cannon['power'] + self.LASER_POWER_STEP)
        elif movement == 2: # Down
            cannon['power'] = max(0.1, cannon['power'] - self.LASER_POWER_STEP)
        elif movement == 3: # Left
            cannon['angle'] -= self.LASER_ROTATION_SPEED
        elif movement == 4: # Right
            cannon['angle'] += self.LASER_ROTATION_SPEED
        cannon['angle'] %= 360

        # Space: Fire laser (on rising edge)
        if space_held and not self.prev_space_held:
            # Sound: Laser Fire
            self._fire_laser(self.selected_cannon_idx)
        self.prev_space_held = space_held

        # Shift: Cycle cannon (on rising edge)
        if shift_held and not self.prev_shift_held:
            # Sound: UI Select
            self.selected_cannon_idx = (self.selected_cannon_idx + 1) % len(self.cannons)
        self.prev_shift_held = shift_held
        
    def _fire_laser(self, cannon_idx):
        cannon = self.cannons[cannon_idx]
        angle_rad = math.radians(cannon['angle'])
        start_pos = cannon['pos']
        # Laser travels across the whole screen
        end_pos = start_pos + pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * (self.SCREEN_WIDTH * 1.5)

        self.laser_beams.append({
            'start': start_pos,
            'end': end_pos,
            'power': cannon['power'],
            'width': 1 + cannon['power'] * 6,
            'lifetime': 5 # a few frames for visual effect
        })

    def _update_projectiles(self):
        for p in self.projectiles:
            if p['active']:
                p['pos'] += p['vel']
                # Check if projectile has passed
                if not self.screen.get_rect().inflate(50, 50).collidepoint(p['pos']):
                    p['active'] = False
                    self.projectiles_passed += 1

    def _update_laser_beams(self):
        self.laser_beams = [beam for beam in self.laser_beams if beam['lifetime'] > 0]
        for beam in self.laser_beams:
            beam['lifetime'] -= 1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] -= 0.2

    def _process_collisions(self):
        hits_this_frame = {i: [] for i, p in enumerate(self.projectiles) if p['active']}
        
        for beam in self.laser_beams:
            if beam['lifetime'] == 4: # Only check for hits on the first frame of the beam
                for i, p in enumerate(self.projectiles):
                    if p['active']:
                        if self._line_circle_collision(beam['start'], beam['end'], p['pos'], p['radius']):
                            hits_this_frame[i].append(beam)

        hit_events = []
        chain_events = []
        
        projectiles_hit_this_step = set()

        for proj_idx, beams in hits_this_frame.items():
            if beams:
                projectiles_hit_this_step.add(proj_idx)
                combined_power = sum(b['power'] for b in beams)
                
                hit_events.append({
                    'proj_idx': proj_idx,
                    'pos': self.projectiles[proj_idx]['pos'],
                    'power': combined_power
                })
                
                if combined_power >= self.CHAIN_REACTION_POWER_THRESHOLD:
                    chained_indices = []
                    for other_idx, other_p in enumerate(self.projectiles):
                        if other_p['active'] and other_idx != proj_idx and other_idx not in projectiles_hit_this_step:
                            dist = self.projectiles[proj_idx]['pos'].distance_to(other_p['pos'])
                            if dist < self.CHAIN_REACTION_RADIUS:
                                chained_indices.append(other_idx)
                                projectiles_hit_this_step.add(other_idx)
                    
                    if chained_indices:
                        chain_events.append({
                            'origin_proj_idx': proj_idx,
                            'origin_pos': self.projectiles[proj_idx]['pos'],
                            'chained_indices': chained_indices
                        })

        return hit_events, chain_events

    def _line_circle_collision(self, p1, p2, circle_center, circle_radius):
        # distance from circle center to the line segment
        p1 = pygame.Vector2(p1)
        p2 = pygame.Vector2(p2)
        circle_center = pygame.Vector2(circle_center)
        
        line_vec = p2 - p1
        if line_vec.length() == 0: return False
        
        t = ((circle_center.x - p1.x) * line_vec.x + (circle_center.y - p1.y) * line_vec.y) / line_vec.length_squared()
        t = max(0, min(1, t))
        
        closest_point = p1 + t * line_vec
        distance = closest_point.distance_to(circle_center)
        
        return distance <= circle_radius

    def _create_explosion(self, pos):
        num_particles = int(20 + self.np_random.uniform(0, 15))
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': self.np_random.uniform(2, 6),
                'lifetime': self.np_random.integers(10, 25),
                'color': self.COLOR_EXPLOSION
            })

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
            "projectiles_intercepted": self.projectiles_intercepted,
            "projectiles_remaining": self.TOTAL_PROJECTILES - (self.projectiles_intercepted + self.projectiles_passed),
        }

    def _render_game(self):
        # Draw grid
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        # Draw particles
        for p in self.particles:
            if p['radius'] > 0:
                alpha = int(255 * (p['lifetime'] / 25))
                color = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color
                )

        # Draw laser beams
        for beam in self.laser_beams:
            alpha = max(0, int(255 * (beam['lifetime'] / 5)))
            # Glow effect
            pygame.draw.aaline(
                self.screen, (*self.COLOR_LASER_GLOW, int(alpha/2)), beam['start'], beam['end'], int(beam['width'] * 1.5)
            )
            # Core beam
            pygame.draw.aaline(
                self.screen, (*self.COLOR_LASER, alpha), beam['start'], beam['end'], int(beam['width'])
            )

        # Draw projectiles
        for p in self.projectiles:
            if p['active']:
                pos_int = (int(p['pos'].x), int(p['pos'].y))
                # Glow
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], p['radius'] + 3, self.COLOR_PROJECTILE_GLOW)
                # Core
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], p['radius'], self.COLOR_PROJECTILE)
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], p['radius'], self.COLOR_PROJECTILE)

        # Draw cannons and aiming lines
        for i, cannon in enumerate(self.cannons):
            pos_int = (int(cannon['pos'].x), int(cannon['pos'].y))
            is_selected = i == self.selected_cannon_idx
            
            # Cannon body
            color = self.COLOR_CANNON_ACTIVE if is_selected else self.COLOR_CANNON_INACTIVE
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.CANNON_SIZE, color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.CANNON_SIZE, color)

            # Aiming line for selected cannon
            if is_selected:
                angle_rad = math.radians(cannon['angle'])
                end_pos = cannon['pos'] + pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * (self.CANNON_SIZE + 10)
                power_alpha = 50 + int(150 * cannon['power'])
                pygame.draw.aaline(self.screen, (*self.COLOR_LASER, power_alpha), cannon['pos'], end_pos, 2)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Projectiles remaining
        remaining = self.TOTAL_PROJECTILES - (self.projectiles_intercepted + self.projectiles_passed)
        proj_text = self.font_large.render(f"TARGETS: {max(0, remaining)}", True, self.COLOR_TEXT)
        self.screen.blit(proj_text, (self.SCREEN_WIDTH - proj_text.get_width() - 10, 10))

        # Selected cannon power bar
        selected_cannon = self.cannons[self.selected_cannon_idx]
        power_bar_width = 200
        power_bar_height = 20
        power_bar_x = self.CENTER[0] - power_bar_width // 2
        power_bar_y = self.SCREEN_HEIGHT - 35
        
        # Background
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (power_bar_x, power_bar_y, power_bar_width, power_bar_height), border_radius=4)
        # Fill
        fill_width = power_bar_width * selected_cannon['power']
        pygame.draw.rect(self.screen, self.COLOR_UI_POWER, (power_bar_x, power_bar_y, fill_width, power_bar_height), border_radius=4)
        # Text
        power_text = self.font_small.render(f"CANNON {self.selected_cannon_idx + 1} POWER", True, self.COLOR_TEXT)
        self.screen.blit(power_text, (power_bar_x + power_bar_width/2 - power_text.get_width()/2, power_bar_y - 20))

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # For manual play, we need a real display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Laser Interceptor")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        space = 0 # 0=released, 1=held
        shift = 0 # 0=released, 1=held
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'R' key
                    obs, info = env.reset()
                    total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)

    env.close()