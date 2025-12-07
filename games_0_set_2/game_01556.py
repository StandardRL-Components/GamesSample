
# Generated: 2025-08-27T17:30:54.693126
# Source Brief: brief_01556.md
# Brief Index: 1556

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↓/↑ to select a gear. ←/→ to rotate the selected gear."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A steampunk puzzle. Rotate the gears to power all four machines before time runs out or the system overloads."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 3000 # 100 seconds at 30 FPS
        self.TIME_LIMIT_SECONDS = 90
        self.MAX_OVERLOADS = 3
        self.ROTATION_INCREMENT = 0.5
        self.MAX_GEAR_SPEED = 10.0

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GEAR_BODY = (218, 165, 32) # Gold
        self.COLOR_GEAR_OUTLINE = (160, 120, 20)
        self.COLOR_ROD = (192, 192, 192) # Silver
        self.COLOR_UI_TEXT = (173, 216, 230) # Light Blue
        self.COLOR_POWERED = (50, 205, 50) # Lime Green
        self.COLOR_UNPOWERED = (70, 70, 70)
        self.COLOR_OVERLOAD = (255, 0, 0) # Red
        self.COLOR_SELECT_GLOW = (255, 255, 0) # Yellow

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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # Initialize state variables
        self.gears = []
        self.machines = []
        self.connections = []
        self.particles = []
        self.solution_speeds = {}
        self.controllable_gear_indices = []

        self.reset()
        
        # self.validate_implementation() # Call this to verify

    def _define_puzzle(self):
        """Defines the gear layout, connections, and machine requirements."""
        self.gears = [
            {'pos': (120, 200), 'radius': 60, 'teeth': 12, 'angle': 0, 'speed': 0, 'target_speed': 0}, # 0: Controllable
            {'pos': (230, 135), 'radius': 40, 'teeth': 8, 'angle': 0, 'speed': 0, 'target_speed': 0},  # 1
            {'pos': (230, 265), 'radius': 40, 'teeth': 8, 'angle': 0, 'speed': 0, 'target_speed': 0},  # 2
            {'pos': (340, 200), 'radius': 60, 'teeth': 12, 'angle': 0, 'speed': 0, 'target_speed': 0}, # 3: Controllable
            {'pos': (450, 135), 'radius': 40, 'teeth': 8, 'angle': 0, 'speed': 0, 'target_speed': 0},  # 4
            {'pos': (450, 265), 'radius': 40, 'teeth': 8, 'angle': 0, 'speed': 0, 'target_speed': 0},  # 5
        ]
        self.controllable_gear_indices = [0, 3]

        # Connections define how gears drive each other: (driver_idx, driven_idx)
        self.connections = [
            (0, 1), (0, 2),
            (3, 4), (3, 5)
        ]

        # Machines are powered by specific gears at target speeds
        self.machines = [
            {'pos': (30, 50), 'gear_idx': 1, 'target_speed': 4.5, 'powered': False},
            {'pos': (30, 350), 'gear_idx': 2, 'target_speed': -4.5, 'powered': False},
            {'pos': (610, 50), 'gear_idx': 4, 'target_speed': -6.0, 'powered': False},
            {'pos': (610, 350), 'gear_idx': 5, 'target_speed': 6.0, 'powered': False},
        ]
        
        # Solution state for reward calculation
        self.solution_speeds = {m['gear_idx']: m['target_speed'] for m in self.machines}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._define_puzzle()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.overloads = 0
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS
        
        self.selected_gear_internal_idx = 0 # index into controllable_gear_indices
        self.particles = []
        self.last_closeness_score = self._calculate_closeness_score()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        reward = 0
        
        if not self.game_over:
            # 1. Handle player input
            selected_gear_idx = self.controllable_gear_indices[self.selected_gear_internal_idx]
            
            if movement == 1: # Up -> Previous Gear
                self.selected_gear_internal_idx = (self.selected_gear_internal_idx - 1) % len(self.controllable_gear_indices)
            elif movement == 2: # Down -> Next Gear
                self.selected_gear_internal_idx = (self.selected_gear_internal_idx + 1) % len(self.controllable_gear_indices)
            elif movement == 3: # Left -> Rotate CCW
                self.gears[selected_gear_idx]['target_speed'] -= self.ROTATION_INCREMENT
            elif movement == 4: # Right -> Rotate CW
                self.gears[selected_gear_idx]['target_speed'] += self.ROTATION_INCREMENT
            
            self.gears[selected_gear_idx]['target_speed'] = np.clip(
                self.gears[selected_gear_idx]['target_speed'], -self.MAX_GEAR_SPEED, self.MAX_GEAR_SPEED
            )
            
            # 2. Update game state
            self._update_gears()
            self._update_particles()
            self.time_left -= 1
            
            # 3. Check for events and calculate rewards
            newly_powered, newly_overloaded = self._check_events()
            reward += newly_powered * 10.0
            if newly_overloaded:
                # Sfx: electric crackle
                self.overloads += 1
                reward -= 5.0

            # Continuous reward for getting closer to solution
            current_closeness_score = self._calculate_closeness_score()
            reward += (current_closeness_score - self.last_closeness_score) * 0.1
            self.last_closeness_score = current_closeness_score
            self.score += reward

        # 4. Check for termination
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.win:
                # Sfx: victory fanfare
                self.score += 100
                reward += 100
            else:
                # Sfx: failure sound
                self.score -= 50
                reward -= 50

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_gears(self):
        # Lerp all gears towards their target speeds for smooth acceleration
        for gear in self.gears:
            gear['speed'] = gear['speed'] * 0.9 + gear['target_speed'] * 0.1

        # Propagate rotation through connections
        for driver_idx, driven_idx in self.connections:
            driver_gear = self.gears[driver_idx]
            driven_gear = self.gears[driven_idx]
            
            ratio = driver_gear['radius'] / driven_gear['radius']
            # Driven gear's target speed is determined by the driver
            driven_gear['target_speed'] = -driver_gear['speed'] * ratio
        
        # Update angles based on current speed
        for gear in self.gears:
            gear['angle'] += gear['speed']
    
    def _calculate_closeness_score(self):
        """Calculates a score based on how close machine gears are to their target speeds."""
        total_closeness = 0
        for m in self.machines:
            gear = self.gears[m['gear_idx']]
            target_speed = m['target_speed']
            if abs(target_speed) > 1e-6:
                error = abs(gear['speed'] - target_speed) / abs(target_speed)
                closeness = max(0, 1 - error)
                total_closeness += closeness
        return total_closeness

    def _check_events(self):
        """Checks for machines being powered and gears overloading. Returns (newly_powered_count, did_overload)."""
        newly_powered_count = 0
        did_overload = False

        # Check machines
        for m in self.machines:
            gear = self.gears[m['gear_idx']]
            is_now_powered = abs(gear['speed'] - m['target_speed']) < 0.5
            if is_now_powered and not m['powered']:
                # Sfx: power up chime
                newly_powered_count += 1
            m['powered'] = is_now_powered
        
        # Check for overloads
        for i, gear in enumerate(self.gears):
            if abs(gear['speed']) > self.MAX_GEAR_SPEED:
                did_overload = True
                self._add_sparks(gear['pos'], 20)
                # Reset speed of overloaded gear and its driver to prevent continuous overload
                gear['target_speed'] = 0
                for d, dr in self.connections:
                    if dr == i:
                        self.gears[d]['target_speed'] *= 0.5 # Dampen the driver
        
        return newly_powered_count, did_overload

    def _check_termination(self):
        if self.win: return True # Already won in a previous step
        
        if all(m['powered'] for m in self.machines):
            self.win = True
            return True
        if self.overloads >= self.MAX_OVERLOADS:
            return True
        if self.time_left <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
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
            "time_left": self.time_left / self.FPS,
            "overloads": self.overloads,
            "powered_machines": sum(1 for m in self.machines if m['powered'])
        }
        
    def _render_game(self):
        # Draw connecting rods (underneath gears)
        for driver_idx, driven_idx in self.connections:
            g1 = self.gears[driver_idx]
            g2 = self.gears[driven_idx]
            pygame.draw.aaline(self.screen, self.COLOR_ROD, g1['pos'], g2['pos'], 2)

        # Draw gears
        selected_gear_idx = self.controllable_gear_indices[self.selected_gear_internal_idx]
        for i, gear in enumerate(self.gears):
            is_selected = (i == selected_gear_idx and not self.game_over)
            self._draw_gear(
                self.screen, 
                gear['pos'], 
                gear['radius'], 
                gear['teeth'], 
                gear['angle'], 
                self.COLOR_GEAR_BODY, 
                self.COLOR_GEAR_OUTLINE,
                is_selected
            )

        # Draw machines
        for m in self.machines:
            color = self.COLOR_POWERED if m['powered'] else self.COLOR_UNPOWERED
            pygame.draw.rect(self.screen, color, (m['pos'][0] - 15, m['pos'][1] - 15, 30, 30))
            pygame.draw.rect(self.screen, self.COLOR_GEAR_OUTLINE, (m['pos'][0] - 15, m['pos'][1] - 15, 30, 30), 2)
            if m['powered']:
                pygame.gfxdraw.filled_circle(self.screen, int(m['pos'][0]), int(m['pos'][1]), 20, (*self.COLOR_POWERED, 30))

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['life']))

    def _draw_gear(self, surface, pos, radius, num_teeth, angle, color_body, color_outline, is_selected):
        x, y = int(pos[0]), int(pos[1])
        
        # Draw selection glow
        if is_selected:
            glow_radius = radius + 10 + 3 * math.sin(self.steps * 0.2)
            pygame.gfxdraw.filled_circle(surface, x, y, int(glow_radius), (*self.COLOR_SELECT_GLOW, 30))
            pygame.gfxdraw.aacircle(surface, x, y, int(glow_radius), (*self.COLOR_SELECT_GLOW, 80))

        # Draw gear teeth
        tooth_width = math.pi * 2 * radius / (num_teeth * 2)
        tooth_height = radius * 0.2
        outer_radius = radius + tooth_height
        
        points = []
        for i in range(num_teeth * 2):
            current_radius = radius if i % 2 == 0 else outer_radius
            current_angle_rad = math.radians(angle) + (i / (num_teeth * 2)) * math.pi * 2
            px = x + current_radius * math.cos(current_angle_rad)
            py = y + current_radius * math.sin(current_angle_rad)
            points.append((px, py))

        pygame.gfxdraw.aapolygon(surface, points, color_outline)
        pygame.gfxdraw.filled_polygon(surface, points, color_body)
        
        # Draw central hub
        inner_radius = int(radius * 0.8)
        pygame.gfxdraw.filled_circle(surface, x, y, inner_radius, color_body)
        pygame.gfxdraw.aacircle(surface, x, y, inner_radius, color_outline)
        pygame.gfxdraw.filled_circle(surface, x, y, int(radius*0.2), color_outline)

    def _add_sparks(self, pos, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.uniform(2, 5),
                'color': random.choice([self.COLOR_OVERLOAD, (255, 165, 0), (255, 255, 0)])
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 0.2
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_ui(self):
        # Timer
        time_str = f"TIME: {max(0, self.time_left // self.FPS):02d}"
        time_surf = self.font_large.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))

        # Overloads
        overload_str = f"OVERLOADS: {self.overloads}/{self.MAX_OVERLOADS}"
        overload_surf = self.font_small.render(overload_str, True, self.COLOR_OVERLOAD)
        self.screen.blit(overload_surf, (10, self.HEIGHT - overload_surf.get_height() - 10))

        # Machine status indicators
        for i, m in enumerate(self.machines):
            color = self.COLOR_POWERED if m['powered'] else self.COLOR_UNPOWERED
            pygame.draw.rect(self.screen, color, (self.WIDTH // 2 - 55 + i * 30, 10, 20, 20))
            pygame.draw.rect(self.screen, self.COLOR_GEAR_OUTLINE, (self.WIDTH // 2 - 55 + i * 30, 10, 20, 20), 1)

        # Game Over / Win message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            message = "SYSTEM ONLINE" if self.win else "SYSTEM FAILURE"
            color = self.COLOR_POWERED if self.win else self.COLOR_OVERLOAD
            msg_surf = self.font_large.render(message, True, color)
            self.screen.blit(msg_surf, (self.WIDTH // 2 - msg_surf.get_width() // 2, self.HEIGHT // 2 - msg_surf.get_height() // 2))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To play manually, you need a pygame window.
    # The environment itself is headless, but we can display its output.
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gear Puzzle Environment")
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_r]: # Reset key
            obs, info = env.reset()
            done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()