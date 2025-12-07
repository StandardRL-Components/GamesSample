import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:00:41.169311
# Source Brief: brief_00216.md
# Brief Index: 216
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control the direction of gravity to merge orbs. Create a massive black hole and use it "
        "to consume all remaining orbs before time runs out."
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to change the direction of gravity and guide the orbs."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_TIME_SECONDS = 45
    
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_BOUNDARY = (200, 200, 220)
    COLOR_TEXT = (230, 230, 240)
    COLOR_TIMER_WARN = (255, 100, 100)
    COLOR_BH_CORE = (0, 0, 0)
    COLOR_BH_GLOW = (100, 80, 200)
    COLOR_GRAVITY_INDICATOR = (255, 255, 255, 150)

    # Physics & Gameplay
    GRAVITY_STRENGTH = 0.05
    DRAG = 0.99
    BOUNCE_DAMPENING = 0.7
    ORB_SPAWN_INTERVAL_SECONDS = 1.0
    INITIAL_ORBS = 3
    BLACK_HOLE_MASS_THRESHOLD = 50
    MIN_ORB_MASS, MAX_ORB_MASS = 5, 15

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
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # Initialize state variables
        self.orbs = []
        self.black_hole = None
        self.particles = []
        self.gravity = np.array([0.0, 0.0])
        self.steps = 0
        self.time_remaining = 0
        self.score = 0
        self.orb_spawn_timer = 0
        self.black_hole_created = False
        
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # this is for dev, not needed in production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.time_remaining = self.MAX_TIME_SECONDS
        self.orb_spawn_timer = 0
        
        self.gravity = np.array([0.0, self.GRAVITY_STRENGTH]) # Start with gravity down
        self.orbs = []
        self.black_hole = None
        self.particles = []
        self.black_hole_created = False

        for _ in range(self.INITIAL_ORBS):
            self._spawn_orb()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # --- 1. Handle Action ---
        movement = action[0]
        # space_held = action[1] == 1 # Unused in this design
        # shift_held = action[2] == 1 # Unused in this design
        
        if movement == 1: # Up
            self.gravity = np.array([0.0, -self.GRAVITY_STRENGTH])
        elif movement == 2: # Down
            self.gravity = np.array([0.0, self.GRAVITY_STRENGTH])
        elif movement == 3: # Left
            self.gravity = np.array([-self.GRAVITY_STRENGTH, 0.0])
        elif movement == 4: # Right
            self.gravity = np.array([self.GRAVITY_STRENGTH, 0.0])
        # movement == 0 is no-op, gravity remains unchanged

        # --- 2. Update Timers and Spawners ---
        self.steps += 1
        self.time_remaining -= 1.0 / self.FPS
        self.orb_spawn_timer += 1.0 / self.FPS
        
        if self.orb_spawn_timer >= self.ORB_SPAWN_INTERVAL_SECONDS:
            self._spawn_orb()
            self.orb_spawn_timer = 0

        # --- 3. Update Game Logic ---
        self._update_entities()
        reward += self._handle_interactions()
        self._update_particles()
        
        # --- 4. Check Termination Conditions ---
        terminated = False
        if self.black_hole:
            bh_pos = self.black_hole['pos']
            if bh_pos[0] < 0 or bh_pos[0] > self.WIDTH or bh_pos[1] < 0 or bh_pos[1] > self.HEIGHT:
                terminated = True
                reward -= 100  # Lose: black hole escaped
        
        if self.time_remaining <= 0 and not terminated:
            terminated = True
            reward -= 100 # Lose: time out
        
        if self.black_hole and not self.orbs and not terminated:
            terminated = True
            reward += 100 # Win: all orbs absorbed
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

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
            "orbs_remaining": len(self.orbs),
            "black_hole_mass": self.black_hole['mass'] if self.black_hole else 0
        }
        
    # --- Game Logic Helpers ---

    def _spawn_orb(self):
        mass = self.np_random.integers(self.MIN_ORB_MASS, self.MAX_ORB_MASS + 1)
        radius = self._mass_to_radius(mass)
        pos = np.array([
            self.np_random.uniform(radius, self.WIDTH - radius),
            self.np_random.uniform(radius, self.HEIGHT - radius)
        ])
        vel = np.array([0.0, 0.0])
        color = self._mass_to_color(mass)
        self.orbs.append({'pos': pos, 'vel': vel, 'mass': mass, 'radius': radius, 'color': color})

    def _update_entities(self):
        entities = self.orbs + ([self.black_hole] if self.black_hole else [])
        for entity in entities:
            entity['vel'] += self.gravity
            entity['vel'] *= self.DRAG
            entity['pos'] += entity['vel']

            # Boundary collision
            if entity['pos'][0] - entity['radius'] < 0:
                entity['pos'][0] = entity['radius']
                entity['vel'][0] *= -self.BOUNCE_DAMPENING
            elif entity['pos'][0] + entity['radius'] > self.WIDTH:
                entity['pos'][0] = self.WIDTH - entity['radius']
                entity['vel'][0] *= -self.BOUNCE_DAMPENING
            
            if entity['pos'][1] - entity['radius'] < 0:
                entity['pos'][1] = entity['radius']
                entity['vel'][1] *= -self.BOUNCE_DAMPENING
            elif entity['pos'][1] + entity['radius'] > self.HEIGHT:
                entity['pos'][1] = self.HEIGHT - entity['radius']
                entity['vel'][1] *= -self.BOUNCE_DAMPENING

    def _handle_interactions(self):
        reward = 0
        
        # Orb-Orb Merges
        merged_indices = set()
        for i in range(len(self.orbs)):
            for j in range(i + 1, len(self.orbs)):
                if i in merged_indices or j in merged_indices:
                    continue
                
                orb1, orb2 = self.orbs[i], self.orbs[j]
                dist_sq = np.sum((orb1['pos'] - orb2['pos'])**2)
                if dist_sq < (orb1['radius'] + orb2['radius'])**2:
                    # Merge orbs
                    total_mass = orb1['mass'] + orb2['mass']
                    # Momentum conservation
                    new_pos = (orb1['pos'] * orb1['mass'] + orb2['pos'] * orb2['mass']) / total_mass
                    new_vel = (orb1['vel'] * orb1['mass'] + orb2['vel'] * orb2['mass']) / total_mass
                    
                    new_orb = {
                        'mass': total_mass,
                        'pos': new_pos,
                        'vel': new_vel,
                        'radius': self._mass_to_radius(total_mass),
                        'color': self._mass_to_color(total_mass)
                    }
                    
                    if not self.black_hole and new_orb['mass'] >= self.BLACK_HOLE_MASS_THRESHOLD:
                        self._create_black_hole(new_orb)
                        reward += 5.0 # Black hole creation reward
                    else:
                        self.orbs.append(new_orb)

                    self._create_particles(new_pos, new_orb['color'], 20, (1, 3), (20, 40))
                    
                    merged_indices.add(i)
                    merged_indices.add(j)
                    reward += 0.1 # Merge reward
        
        # Remove merged orbs
        if merged_indices:
            self.orbs = [orb for i, orb in enumerate(self.orbs) if i not in merged_indices]

        # Black Hole Absorption
        if self.black_hole:
            absorbed_indices = set()
            for i, orb in enumerate(self.orbs):
                dist_sq = np.sum((self.black_hole['pos'] - orb['pos'])**2)
                if dist_sq < (self.black_hole['radius'] + orb['radius'])**2:
                    # Absorb orb
                    total_mass = self.black_hole['mass'] + orb['mass']
                    # Momentum conservation
                    self.black_hole['vel'] = (self.black_hole['vel'] * self.black_hole['mass'] + orb['vel'] * orb['mass']) / total_mass
                    self.black_hole['mass'] = total_mass
                    self.black_hole['radius'] = self._mass_to_radius(total_mass)
                    
                    self._create_particles(orb['pos'], orb['color'], 30, (2, 5), (30, 60))
                    
                    absorbed_indices.add(i)
                    reward += 1.0 # Absorption reward
            
            if absorbed_indices:
                self.orbs = [orb for i, orb in enumerate(self.orbs) if i not in absorbed_indices]
                
        return reward

    def _create_black_hole(self, source_orb):
        self.black_hole = {
            'pos': source_orb['pos'],
            'vel': source_orb['vel'],
            'mass': source_orb['mass'],
            'radius': self._mass_to_radius(source_orb['mass']),
            'angle': 0
        }
        self.black_hole_created = True
        self._create_particles(source_orb['pos'], self.COLOR_BH_GLOW, 100, (2, 6), (60, 120))
    
    def _create_particles(self, pos, color, count, speed_range, life_range):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(*speed_range)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifespan = self.np_random.integers(life_range[0], life_range[1])
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'color': color})

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Particle drag
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles
        
    # --- Rendering Helpers ---

    def _render_game(self):
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            # Pygame rects don't support alpha, so we skip drawing if transparent
            if alpha > 0:
                color = (*p['color'], alpha)
                size = max(1, int(3 * (p['life'] / p['max_life'])))
                
                # Create a temporary surface for alpha blending
                particle_surf = pygame.Surface((size, size), pygame.SRCALPHA)
                particle_surf.fill((*p['color'], alpha))
                self.screen.blit(particle_surf, p['pos'])

        # Orbs
        for orb in self.orbs:
            pos_int = orb['pos'].astype(int)
            radius_int = int(orb['radius'])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius_int, orb['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius_int, orb['color'])
            self._render_text(str(orb['mass']), pos_int, 14, self.COLOR_BG)

        # Black Hole
        if self.black_hole:
            self._render_accretion_disk()
            pos_int = self.black_hole['pos'].astype(int)
            radius_int = int(self.black_hole['radius'])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius_int, self.COLOR_BH_CORE)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius_int, self.COLOR_BH_CORE)

        # Gravity Indicator
        self._render_gravity_indicator()

        # Boundary
        pygame.draw.rect(self.screen, self.COLOR_BOUNDARY, (0, 0, self.WIDTH, self.HEIGHT), 2)
        
    def _render_accretion_disk(self):
        if not self.black_hole: return
        self.black_hole['angle'] = (self.black_hole['angle'] + 2) % 360
        pos = self.black_hole['pos']
        base_radius = self.black_hole['radius'] * 1.5
        
        for i in range(3):
            radius = int(base_radius * (1 + i * 0.25))
            alpha = 150 - i * 40
            angle_offset = (i * 45)
            
            # Create a temporary surface for the rotating ring
            ring_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            
            # Draw arcs to create a broken ring effect
            for j in range(3):
                start_angle = math.radians(self.black_hole['angle'] + angle_offset + j * 120)
                end_angle = math.radians(self.black_hole['angle'] + angle_offset + j * 120 + 80)
                pygame.draw.arc(ring_surface, (*self.COLOR_BH_GLOW, alpha), (0, 0, radius*2, radius*2), start_angle, end_angle, width=max(2, int(base_radius * 0.1)))

            self.screen.blit(ring_surface, (pos[0] - radius, pos[1] - radius))

    def _render_gravity_indicator(self):
        center = (self.WIDTH - 40, 40)
        if np.linalg.norm(self.gravity) > 0:
            direction = self.gravity / np.linalg.norm(self.gravity)
        else:
            direction = np.array([0, 0])
        
        p1 = center + direction * 15
        p2 = center - direction * 5 + np.array([-direction[1], direction[0]]) * 8
        p3 = center - direction * 5 - np.array([-direction[1], direction[0]]) * 8
        
        pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_GRAVITY_INDICATOR)
        pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_GRAVITY_INDICATOR)

    def _render_ui(self):
        # Score
        self._render_text(f"SCORE: {int(self.score)}", (10, 10), 16, self.COLOR_TEXT, align="topleft")
        # Orbs remaining
        self._render_text(f"ORBS: {len(self.orbs)}", (10, 30), 16, self.COLOR_TEXT, align="topleft")
        
        # Timer
        timer_color = self.COLOR_TEXT if self.time_remaining > 10 else self.COLOR_TIMER_WARN
        time_text = f"{max(0, self.time_remaining):.1f}"
        self._render_text(time_text, (self.WIDTH - 10, 10), 32, timer_color, align="topright")

    def _render_text(self, text, pos, size, color, align="center"):
        font = self.font_small if size <= 16 else self.font_large
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "topleft":
            text_rect.topleft = pos
        elif align == "topright":
            text_rect.topright = pos
        self.screen.blit(text_surface, text_rect)

    # --- Utility Methods ---

    def _mass_to_radius(self, mass):
        # Radius proportional to sqrt of mass (area)
        return max(5, int(3 * math.sqrt(mass)))

    def _mass_to_color(self, mass):
        # Hue cycles with mass for visual variety
        hue = int((mass * 10) % 360)
        color = pygame.Color(0)
        color.hsva = (hue, 80, 95, 100)
        return (color.r, color.g, color.b)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to run and visualize the environment
if __name__ == '__main__':
    # This block will not run in the testing environment, but is useful for human interaction.
    # It requires a display, so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Gravity Orb Merger")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # --- Human Controls ---
    # ARROW KEYS to change gravity
    # R to reset
    # Q to quit
    
    action = np.array([0, 0, 0]) # Start with no-op
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    env.close()
                    quit()
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
                
                # Map keys to actions
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
            
            if event.type == pygame.KEYUP:
                # Reset to no-op when key is released
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    action[0] = 0

        if terminated:
            # On termination, just wait for reset
            action[0] = 0 # No-op
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)