import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:43:29.157083
# Source Brief: brief_02765.md
# Brief Index: 2765
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
        "Defend your workshop from an onslaught of steampunk robots. Place and upgrade barricades to hold the line and survive until the timer runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Tap space to switch barricade types, or hold to charge and release to place. Press shift to activate a time freeze."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    FPS = 30

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GEAR = (15, 25, 35)
    COLOR_WORKSHOP = (10, 15, 20)
    COLOR_ROBOT = (220, 50, 50)
    COLOR_ROBOT_EYE = (255, 255, 255)
    COLOR_ROBOT_GLOW = (220, 50, 50, 50)
    COLOR_BOLT = (0, 150, 255)
    COLOR_BOLT_UPGRADED = (0, 220, 255)
    COLOR_BOLT_GLOW = (0, 150, 255, 50)
    COLOR_SPRING = (50, 200, 50)
    COLOR_SPRING_UPGRADED = (100, 255, 100)
    COLOR_SPRING_GLOW = (50, 200, 50, 50)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_UI_BG = (0, 0, 0, 128)
    COLOR_FREEZE_OVERLAY = (150, 100, 255, 100)
    
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
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)

        self.render_mode = render_mode
        self._gears = self._create_gears()
        
        # This call initializes state variables
        # self.reset() is called by the agent/wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        # Player/Cursor state
        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        self.cursor_speed = 10.0
        self.selected_barricade_type = 'bolt'
        self.placing_barricade = None
        self.space_press_duration = 0
        
        # Entities
        self.robots = []
        self.barricades = []
        self.particles = []

        # Resources & Timers
        self.resources = {'bolt': 10, 'spring': 5, 'freeze': 1}
        self.time_freeze_timer = 0
        self.time_freeze_active = False
        self.time_freeze_duration = 5 * self.FPS  # 5 seconds
        self.kills_during_freeze = 0
        
        self.robot_spawn_cooldown = 0
        self.robot_spawn_interval = 2 * self.FPS # 2 seconds
        
        # Unlocks
        self.unlocks = {'strong_bolt': False, 'fast_spring': False, 'extra_freeze': False}

        # Input tracking
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # Game boundaries
        self.workshop_boundary_x = 40

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        terminated = False
        truncated = False
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held, shift_held)
        
        # --- Game Logic Update ---
        self.steps += 1
        
        self._check_unlocks()
        
        reward_bonus = self._update_time_freeze()
        reward += reward_bonus
        
        self._update_spawners()
        
        destroyed_robots_count = self._update_entities()
        reward += destroyed_robots_count * 0.1
        self.score += destroyed_robots_count * 10

        # Continuous negative reward for existing robots
        reward -= 0.001 * len(self.robots)

        # --- Termination Check ---
        if any(r['pos'][0] < self.workshop_boundary_x for r in self.robots):
            terminated = True
            reward -= 100
            self.game_over = True
            # sfx: game_over_sound
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            terminated = True
            reward += 100
            self.game_over = True
            self.win = True
            # sfx: victory_fanfare

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Cursor movement
        if movement == 1: self.cursor_pos[1] -= self.cursor_speed
        elif movement == 2: self.cursor_pos[1] += self.cursor_speed
        elif movement == 3: self.cursor_pos[0] -= self.cursor_speed
        elif movement == 4: self.cursor_pos[0] += self.cursor_speed
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

        # Time Freeze
        shift_pressed = shift_held and not self.prev_shift_held
        if shift_pressed and self.resources['freeze'] > 0 and not self.time_freeze_active:
            self.resources['freeze'] -= 1
            self.time_freeze_active = True
            self.time_freeze_timer = self.time_freeze_duration
            self.kills_during_freeze = 0
            # sfx: time_freeze_activate
            for _ in range(50): # Screen flash effect
                self.particles.append(self._create_particle(
                    pos=np.array([random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT)]),
                    color=self.COLOR_FREEZE_OVERLAY[:3],
                    life=self.FPS // 3,
                    size=random.uniform(10, 30),
                    vel=np.array([0,0])
                ))

        # Barricade Placement & Type Toggle
        space_pressed = space_held and not self.prev_space_held
        space_released = not space_held and self.prev_space_held

        if space_held:
            self.space_press_duration += 1
        
        if space_pressed and self.placing_barricade is None:
            self.placing_barricade = {
                'type': self.selected_barricade_type,
                'pos': self.cursor_pos.copy(),
                'size': 10,
                'max_size': 150,
                'growth_rate': 2,
            }
            # sfx: start_placement_sound
        
        if space_held and self.placing_barricade is not None:
            self.placing_barricade['size'] = min(
                self.placing_barricade['max_size'],
                self.placing_barricade['size'] + self.placing_barricade['growth_rate']
            )

        if space_released:
            if self.space_press_duration <= 4 and self.placing_barricade is not None: # Tap
                self.selected_barricade_type = 'spring' if self.selected_barricade_type == 'bolt' else 'bolt'
                self.placing_barricade = None
                # sfx: toggle_type_sound
            elif self.placing_barricade is not None: # Hold-release
                res_type = self.placing_barricade['type']
                if self.resources[res_type] > 0:
                    self.resources[res_type] -= 1
                    new_barricade = self.placing_barricade.copy()
                    new_barricade['id'] = self.steps + random.random()
                    new_barricade['hp'] = 2 if self.unlocks['strong_bolt'] and res_type == 'bolt' else 1
                    if res_type == 'spring':
                        new_barricade['oscillation_phase'] = random.uniform(0, 2 * math.pi)
                        new_barricade['base_size'] = new_barricade['size']
                        new_barricade['oscillation_speed'] = 0.15 if self.unlocks['fast_spring'] else 0.05
                    self.barricades.append(new_barricade)
                    # sfx: place_barricade_sound
                self.placing_barricade = None
            self.space_press_duration = 0

    def _check_unlocks(self):
        if not self.unlocks['strong_bolt'] and self.steps >= 500:
            self.unlocks['strong_bolt'] = True
            # sfx: unlock_sound
        if not self.unlocks['fast_spring'] and self.steps >= 1000:
            self.unlocks['fast_spring'] = True
            # sfx: unlock_sound
        if not self.unlocks['extra_freeze'] and self.steps >= 1500:
            self.unlocks['extra_freeze'] = True
            self.resources['freeze'] += 1
            # sfx: unlock_sound

    def _update_time_freeze(self):
        reward = 0
        if self.time_freeze_timer > 0:
            self.time_freeze_timer -= 1
            if self.time_freeze_timer == 0:
                self.time_freeze_active = False
                # sfx: time_freeze_deactivate
                if self.kills_during_freeze >= 3:
                    # sfx: bonus_reward_sound
                    reward = 1.0
        return reward

    def _update_spawners(self):
        # Increase spawn rate over time
        if self.steps > 0 and self.steps % 100 == 0:
            self.robot_spawn_interval = max(15, self.robot_spawn_interval - (0.001 * self.FPS * 100))

        if not self.time_freeze_active:
            self.robot_spawn_cooldown -= 1
            if self.robot_spawn_cooldown <= 0:
                self.robot_spawn_cooldown = self.robot_spawn_interval
                self._spawn_robot()

    def _spawn_robot(self):
        self.robots.append({
            'pos': np.array([self.SCREEN_WIDTH + 20, random.uniform(20, self.SCREEN_HEIGHT - 20)], dtype=float),
            'speed': random.uniform(1.0, 1.5),
            'size': 12,
        })

    def _update_entities(self):
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] *= 0.98

        # Update spring barricades
        for b in self.barricades:
            if b['type'] == 'spring':
                b['oscillation_phase'] += b['oscillation_speed']
                oscillation = (math.sin(b['oscillation_phase']) + 1) / 2 # Range 0-1
                b['size'] = b['base_size'] * (0.5 + oscillation * 0.75)

        if self.time_freeze_active:
            return 0
            
        # Update robots
        robots_to_remove = set()
        destroyed_robots_count = 0
        for i, robot in enumerate(self.robots):
            robot['pos'][0] -= robot['speed']
            
            # Check for collisions with barricades
            r_rect = pygame.Rect(robot['pos'][0] - robot['size']/2, robot['pos'][1] - robot['size']/2, robot['size'], robot['size'])
            for b in self.barricades:
                b_rect = pygame.Rect(b['pos'][0] - 4, b['pos'][1] - b['size']/2, 8, b['size'])
                if r_rect.colliderect(b_rect):
                    robots_to_remove.add(i)
                    self._create_explosion(robot['pos'], self.COLOR_ROBOT)
                    # sfx: robot_explosion
                    destroyed_robots_count += 1
                    if self.time_freeze_timer > 0: self.kills_during_freeze += 1
                    
                    b['hp'] -= 1
                    if b['hp'] <= 0:
                        self.barricades = [bar for bar in self.barricades if bar['id'] != b['id']]
                        color = self.COLOR_BOLT if b['type'] == 'bolt' else self.COLOR_SPRING
                        self._create_explosion(b['pos'], color)
                        # sfx: barricade_break
                    break
        
        if robots_to_remove:
            self.robots = [r for i, r in enumerate(self.robots) if i not in robots_to_remove]
        
        return destroyed_robots_count

    def _create_explosion(self, pos, color):
        for _ in range(20):
            self.particles.append(self._create_particle(
                pos=pos.copy(),
                color=color,
                life=random.randint(15, 30),
                size=random.uniform(1, 5),
                vel=np.array([random.uniform(-3, 3), random.uniform(-3, 3)])
            ))

    def _create_particle(self, pos, color, life, size, vel):
        return {'pos': pos, 'color': color, 'life': life, 'max_life': life, 'size': size, 'vel': vel}

    def _get_observation(self):
        self._render_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "resources": self.resources, "win": self.win}

    def _render_frame(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_barricades()
        self._render_robots()
        self._render_particles()
        self._render_cursor()
        if self.time_freeze_active:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_FREEZE_OVERLAY)
            self.screen.blit(overlay, (0,0))
        self._render_ui()
        if self.game_over:
            self._render_game_over()

    def _render_background(self):
        # Workshop area
        pygame.draw.rect(self.screen, self.COLOR_WORKSHOP, (0, 0, self.workshop_boundary_x, self.SCREEN_HEIGHT))
        pygame.draw.line(self.screen, (50, 60, 70), (self.workshop_boundary_x, 0), (self.workshop_boundary_x, self.SCREEN_HEIGHT), 2)
        
        # Decorative gears
        for gear in self._gears:
            pygame.gfxdraw.filled_circle(self.screen, gear['x'], gear['y'], gear['r'], self.COLOR_GEAR)
            pygame.gfxdraw.aacircle(self.screen, gear['x'], gear['y'], gear['r'], self.COLOR_GEAR)

    def _create_gears(self):
        gears = []
        for _ in range(5):
            gears.append({
                'x': random.randint(0, self.SCREEN_WIDTH),
                'y': random.randint(0, self.SCREEN_HEIGHT),
                'r': random.randint(30, 100)
            })
        return gears

    def _render_robots(self):
        for robot in self.robots:
            pos = robot['pos'].astype(int)
            size = int(robot['size'])
            
            # Glow
            glow_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_ROBOT_GLOW, (size, size), size)
            self.screen.blit(glow_surf, (pos[0]-size, pos[1]-size))
            
            # Body
            pygame.draw.rect(self.screen, self.COLOR_ROBOT, (pos[0] - size//2, pos[1] - size//2, size, size))
            
            # Eye
            eye_pos = (pos[0] - size//4, pos[1])
            pygame.draw.circle(self.screen, self.COLOR_ROBOT_EYE, eye_pos, int(size/4))

    def _render_barricades(self):
        for b in self.barricades:
            pos = b['pos'].astype(int)
            size = int(b['size'])
            
            is_bolt = b['type'] == 'bolt'
            is_upgraded = (is_bolt and self.unlocks['strong_bolt']) or (not is_bolt and self.unlocks['fast_spring'])
            
            if is_bolt:
                color = self.COLOR_BOLT_UPGRADED if is_upgraded else self.COLOR_BOLT
                glow_color = self.COLOR_BOLT_GLOW
            else: # Spring
                color = self.COLOR_SPRING_UPGRADED if is_upgraded else self.COLOR_SPRING
                glow_color = self.COLOR_SPRING_GLOW

            rect = pygame.Rect(pos[0] - 4, pos[1] - size//2, 8, size)
            
            # Glow
            glow_surf = pygame.Surface((24, size + 16), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, glow_color, (0, 0, 24, size + 16), border_radius=8)
            self.screen.blit(glow_surf, (rect.x - 8, rect.y - 8))

            # Body
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            # Highlight
            pygame.draw.line(self.screen, (255,255,255,80), (rect.left+2, rect.top+2), (rect.left+2, rect.bottom-2), 2)
            if is_upgraded:
                pygame.draw.rect(self.screen, (255, 215, 0), rect, 1, border_radius=3)


    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pos = p['pos'].astype(int)
            size = max(1, int(p['size']))
            
            surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (size, size), size)
            self.screen.blit(surf, (pos[0]-size, pos[1]-size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_cursor(self):
        pos = self.cursor_pos.astype(int)
        
        # Crosshair
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos[0]-10, pos[1]), (pos[0]+10, pos[1]), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos[0], pos[1]-10), (pos[0], pos[1]+10), 1)

        # Placement preview
        barricade = self.placing_barricade
        if barricade is not None:
            size = int(barricade['size'])
            color = self.COLOR_BOLT if barricade['type'] == 'bolt' else self.COLOR_SPRING
            color_alpha = (*color, 100)
            rect = pygame.Rect(pos[0] - 4, pos[1] - size//2, 8, size)
            
            shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            shape_surf.fill(color_alpha)
            self.screen.blit(shape_surf, rect.topleft)
            pygame.draw.rect(self.screen, color, rect, 1)

    def _render_ui(self):
        # Info Panel
        panel = pygame.Surface((self.SCREEN_WIDTH, 30), pygame.SRCALPHA)
        panel.fill(self.COLOR_UI_BG)
        self.screen.blit(panel, (0, 0))

        # Resources
        bolt_color = self.COLOR_BOLT_UPGRADED if self.unlocks['strong_bolt'] else self.COLOR_BOLT
        spring_color = self.COLOR_SPRING_UPGRADED if self.unlocks['fast_spring'] else self.COLOR_SPRING
        freeze_color = self.COLOR_FREEZE_OVERLAY[:3]

        bolt_text = self.font_small.render(f"Bolts: {self.resources['bolt']}", True, bolt_color)
        spring_text = self.font_small.render(f"Springs: {self.resources['spring']}", True, spring_color)
        freeze_text = self.font_small.render(f"Freeze: {self.resources['freeze']}", True, freeze_color)
        
        self.screen.blit(bolt_text, (10, 7))
        self.screen.blit(spring_text, (120, 7))
        self.screen.blit(freeze_text, (240, 7))

        # Selected type indicator
        sel_type_text = self.selected_barricade_type.upper()
        sel_type_color = self.COLOR_BOLT if self.selected_barricade_type == 'bolt' else self.COLOR_SPRING
        sel_text = self.font_small.render(f"Selected: {sel_type_text}", True, sel_type_color)
        self.screen.blit(sel_text, (350, 7))

        # Score and Time
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        time_text = self.font_small.render(f"Time: {self.MAX_STEPS - self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - 220, 7))
        self.screen.blit(time_text, (self.SCREEN_WIDTH - 110, 7))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        message = "VICTORY" if self.win else "DEFENSES BREACHED"
        color = (100, 255, 100) if self.win else (255, 100, 100)
        
        text = self.font_large.render(message, True, color)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
        self.screen.blit(text, text_rect)
        
        score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
        self.screen.blit(score_text, score_rect)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block is for manual play and is not part of the Gymnasium environment
    # It will not be run by the test suite.
    
    # Un-set the dummy driver to see the display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Steampunk Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # none
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Wait 3 seconds before resetting
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(GameEnv.FPS)
        
    env.close()