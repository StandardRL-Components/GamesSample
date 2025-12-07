import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent deploys and upgrades geometric prisms
    to absorb chaotic distortions, protecting a central grid. The goal is to
    survive for a fixed duration against increasingly difficult waves of distortions.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}
    
    game_description = (
        "Deploy and upgrade geometric prisms to absorb chaotic distortions and protect a central grid against increasingly difficult waves."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to deploy or upgrade a prism. Press shift to cycle between prism types."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Critical Gymnasium Interface ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame & Display Setup ---
        self.render_mode = render_mode
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.display = None
        if self.render_mode == "human":
            self.display = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # --- Visual & Game Constants ---
        self._define_constants()
        
        # --- Fonts ---
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.grid_health = 0.0
        self.energy = 0.0
        self.cursor_pos = [0, 0]
        self.prisms = []
        self.distortions = []
        self.particles = []
        self.selected_prism_type_idx = 0
        self.distortion_spawn_timer = 0
        self.last_space_press = False
        self.last_shift_press = False
        self.current_distortion_speed = 0.0
    
    def _define_constants(self):
        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GRID = (30, 35, 60)
        self.COLOR_CONDUIT = (40, 100, 120)
        self.COLOR_DISTORTION = (255, 50, 80)
        self.COLOR_ENERGY = (255, 220, 50)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_HEALTH = (50, 200, 100)
        self.COLOR_CURSOR = (255, 255, 255)

        # Game Parameters
        self.MAX_STEPS = 1000
        self.INITIAL_HEALTH = 100.0
        self.INITIAL_ENERGY = 50.0
        self.PASSIVE_ENERGY_GAIN = 0.05
        self.ENERGY_PER_ABSORPTION = 5.0
        self.CURSOR_SPEED = 10
        self.DISTORTION_DAMAGE = 5.0
        self.DISTORTION_BASE_SPEED = 1.0
        self.DISTORTION_BASE_SPAWN_INTERVAL = 30 # steps

        # Prism Types
        self.PRISM_TYPES = [
            {
                "name": "Absorber Prism",
                "color": (80, 150, 255),
                "shape": "triangle",
                "cost": 25,
                "upgrade_cost_base": 15,
                "base_radius": 40,
                "energy_bonus_per_level": 0.0,
            },
            {
                "name": "Catalyst Prism",
                "color": (200, 100, 255),
                "shape": "square",
                "cost": 40,
                "upgrade_cost_base": 25,
                "base_radius": 30,
                "energy_bonus_per_level": 1.5,
            }
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.grid_health = self.INITIAL_HEALTH
        self.energy = self.INITIAL_ENERGY
        
        self.cursor_pos = [self.width // 2, self.height // 2]
        self.prisms = []
        self.distortions = []
        self.particles = []
        
        self.selected_prism_type_idx = 0
        self.distortion_spawn_timer = self.DISTORTION_BASE_SPAWN_INTERVAL
        
        self.last_space_press = True # Prevent action on first frame
        self.last_shift_press = True
        self.current_distortion_speed = self.DISTORTION_BASE_SPEED

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # --- Unpack action and handle input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        upgrade_reward = self._handle_input(movement, space_held, shift_held)
        
        # --- Update game logic ---
        self._update_difficulty()
        self._update_energy()
        self._spawn_distortions()
        self._update_prisms()
        absorptions_this_step = self._update_distortions()
        self._update_particles()
        
        # --- Calculate reward ---
        reward = self._calculate_reward(upgrade_reward, absorptions_this_step)
        
        # --- Check for termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.grid_health <= 0:
                reward -= 100.0  # Penalty for losing
            else:  # Survived MAX_STEPS
                reward += 100.0  # Bonus for winning
        
        self.score += reward
        truncated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Cursor movement
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.width)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.height)

        # Cycle prism type (on press)
        if shift_held and not self.last_shift_press:
            self.selected_prism_type_idx = (self.selected_prism_type_idx + 1) % len(self.PRISM_TYPES)
        self.last_shift_press = shift_held

        # Deploy/Upgrade prism (on press)
        upgrade_reward = 0
        if space_held and not self.last_space_press:
            prism_under_cursor = None
            for prism in self.prisms:
                dist = math.hypot(prism['pos'][0] - self.cursor_pos[0], prism['pos'][1] - self.cursor_pos[1])
                if dist < prism['radius'] * 0.5:
                    prism_under_cursor = prism
                    break
            
            if prism_under_cursor:
                upgrade_reward = self._upgrade_prism(prism_under_cursor)
            else:
                self._deploy_prism()
        self.last_space_press = space_held
        
        return upgrade_reward

    def _deploy_prism(self):
        prism_type = self.PRISM_TYPES[self.selected_prism_type_idx]
        if self.energy >= prism_type['cost']:
            self.energy -= prism_type['cost']
            new_prism = {
                'pos': list(self.cursor_pos),
                'type_idx': self.selected_prism_type_idx,
                'level': 1,
                'radius': prism_type['base_radius'],
                'angle': self.np_random.uniform(0, 2 * math.pi),
                'rotation_speed': self.np_random.uniform(0.01, 0.03)
            }
            self.prisms.append(new_prism)

    def _upgrade_prism(self, prism):
        prism_type = self.PRISM_TYPES[prism['type_idx']]
        cost = prism_type['upgrade_cost_base'] * prism['level']
        if self.energy >= cost:
            self.energy -= cost
            prism['level'] += 1
            prism['radius'] = prism_type['base_radius'] * (1 + 0.1 * (prism['level'] - 1))
            return 1.0 # Reward for upgrading
        return 0.0

    def _update_difficulty(self):
        # Spawn rate increases
        self.distortion_spawn_timer -= 1
        # Speed increases
        self.current_distortion_speed = self.DISTORTION_BASE_SPEED + (self.steps / 100) * 0.05

    def _update_energy(self):
        self.energy = min(100.0, self.energy + self.PASSIVE_ENERGY_GAIN)

    def _spawn_distortions(self):
        spawn_interval = self.DISTORTION_BASE_SPAWN_INTERVAL - (self.steps // 50)
        if self.distortion_spawn_timer <= 0:
            self.distortion_spawn_timer = max(5, spawn_interval)
            
            edge = self.np_random.integers(4)
            if edge == 0: # top
                pos = [self.np_random.uniform(0, self.width), -10]
            elif edge == 1: # bottom
                pos = [self.np_random.uniform(0, self.width), self.height + 10]
            elif edge == 2: # left
                pos = [-10, self.np_random.uniform(0, self.height)]
            else: # right
                pos = [self.width + 10, self.np_random.uniform(0, self.height)]
            
            target = [self.np_random.uniform(self.width*0.3, self.width*0.7), 
                      self.np_random.uniform(self.height*0.3, self.height*0.7)]
            angle = math.atan2(target[1] - pos[1], target[0] - pos[0])
            
            new_distortion = {
                'pos': pos,
                'vel': [math.cos(angle), math.sin(angle)],
                'size': self.np_random.uniform(8, 12),
                'pulse': 0
            }
            self.distortions.append(new_distortion)

    def _update_prisms(self):
        for prism in self.prisms:
            prism['angle'] += prism['rotation_speed']

    def _update_distortions(self):
        absorptions = 0
        distortions_to_remove = []
        for i, d in enumerate(self.distortions):
            d['pos'][0] += d['vel'][0] * self.current_distortion_speed
            d['pos'][1] += d['vel'][1] * self.current_distortion_speed
            d['pulse'] += 0.1

            absorbed = False
            for prism in self.prisms:
                dist = math.hypot(d['pos'][0] - prism['pos'][0], d['pos'][1] - prism['pos'][1])
                if dist < prism['radius']:
                    prism_type = self.PRISM_TYPES[prism['type_idx']]
                    energy_gain = self.ENERGY_PER_ABSORPTION + prism_type['energy_bonus_per_level'] * (prism['level'] - 1)
                    self.energy += energy_gain
                    self._create_particles(d['pos'], self.COLOR_ENERGY, 20)
                    distortions_to_remove.append(i)
                    absorptions += 1
                    absorbed = True
                    break
            
            if not absorbed and (d['pos'][0] < -10 or d['pos'][0] > self.width + 10 or d['pos'][1] < -10 or d['pos'][1] > self.height + 10):
                self.grid_health -= self.DISTORTION_DAMAGE
                self._create_particles(d['pos'], self.COLOR_DISTORTION, 10, 2.0)
                distortions_to_remove.append(i)
        
        for i in sorted(list(set(distortions_to_remove)), reverse=True):
            del self.distortions[i]
            
        return absorptions

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'max_life': 30,
                'color': color,
                'radius': self.np_random.uniform(1, 4)
            })

    def _calculate_reward(self, upgrade_reward, absorptions):
        reward = upgrade_reward
        if absorptions > 0:
            reward += 0.1 * absorptions
        else:
            reward -= 0.01 # Small penalty for inaction
        return reward

    def _check_termination(self):
        return self.grid_health <= 0 or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_entities()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_background(self):
        for x in range(0, self.width, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.height))
        for y in range(0, self.height, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.width, y))
        
        for i in range(5):
            start_y = 80 * i
            pygame.gfxdraw.bezier(self.screen, [(0, start_y), (160, start_y + 40), (480, start_y - 40), (640, start_y)], 10, self.COLOR_CONDUIT)

        intensity = min(1.0, self.steps / (self.MAX_STEPS * 0.8))
        edge_color = (
            int(50 + 150 * intensity),
            int(150 - 100 * intensity),
            int(100 - 50 * intensity)
        )
        pygame.draw.rect(self.screen, edge_color, (0, 0, self.width, self.height), 5, border_radius=5)

    def _render_entities(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))

        for d in self.distortions:
            pulse_size = d['size'] + math.sin(d['pulse']) * 2
            pygame.gfxdraw.aacircle(self.screen, int(d['pos'][0]), int(d['pos'][1]), int(pulse_size), self.COLOR_DISTORTION)
            pygame.gfxdraw.filled_circle(self.screen, int(d['pos'][0]), int(d['pos'][1]), int(pulse_size), self.COLOR_DISTORTION)

        for p in self.prisms:
            self._draw_prism(p)
            
    def _draw_prism(self, prism):
        prism_type = self.PRISM_TYPES[prism['type_idx']]
        pos = (int(prism['pos'][0]), int(prism['pos'][1]))
        
        radius_color = (*prism_type['color'], 50)
        temp_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, pos[0], pos[1], int(prism['radius']), radius_color)
        pygame.gfxdraw.aacircle(temp_surf, pos[0], pos[1], int(prism['radius']), (*prism_type['color'], 100))
        self.screen.blit(temp_surf, (0,0))

        size = 10 + prism['level']
        points = []
        num_points = 3 if prism_type['shape'] == 'triangle' else 4
        
        for i in range(num_points):
            angle = prism['angle'] + (2 * math.pi * i / num_points)
            x = pos[0] + size * math.cos(angle)
            y = pos[1] + size * math.sin(angle)
            points.append((int(x), int(y)))
        
        glow_color = (*prism_type['color'], 100)
        pygame.gfxdraw.filled_polygon(self.screen, points, glow_color)
        pygame.gfxdraw.aapolygon(self.screen, points, glow_color)
        
        pygame.gfxdraw.filled_polygon(self.screen, points, prism_type['color'])
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_UI_TEXT)

        level_text = self.font_small.render(f"L{prism['level']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (pos[0] - level_text.get_width()//2, pos[1] - level_text.get_height()//2))

    def _render_ui(self):
        bar_height = 15
        pygame.draw.rect(self.screen, self.COLOR_GRID, (10, 10, 200, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.width - 210, 10, 200, bar_height))
        
        health_width = max(0, int(200 * (self.grid_health / self.INITIAL_HEALTH)))
        energy_width = max(0, int(200 * (self.energy / 100.0)))
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (10, 10, health_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_ENERGY, (self.width - 210, 10, energy_width, bar_height))

        health_text = self.font_small.render("GRID INTEGRITY", True, self.COLOR_UI_TEXT)
        energy_text = self.font_small.render("ENERGY", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 10 + bar_height))
        self.screen.blit(energy_text, (self.width - 205, 10 + bar_height))

        score_text = self.font_medium.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        steps_text = self.font_medium.render(f"Time: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.width//2 - score_text.get_width()//2, 5))
        self.screen.blit(steps_text, (self.width//2 - steps_text.get_width()//2, 30))

        bottom_panel_h = 50
        pygame.draw.rect(self.screen, (self.COLOR_BG[0]+10, self.COLOR_BG[1]+10, self.COLOR_BG[2]+10), (0, self.height - bottom_panel_h, self.width, bottom_panel_h))
        
        prism_type = self.PRISM_TYPES[self.selected_prism_type_idx]
        prism_name = self.font_medium.render(f"Selected: {prism_type['name']}", True, self.COLOR_UI_TEXT)
        prism_cost = self.font_small.render(f"Cost: {prism_type['cost']} E", True, self.COLOR_ENERGY if self.energy >= prism_type['cost'] else self.COLOR_DISTORTION)
        
        self.screen.blit(prism_name, (20, self.height - bottom_panel_h + 5))
        self.screen.blit(prism_cost, (20, self.height - bottom_panel_h + 30))

        shift_text = self.font_small.render("[SHIFT] to cycle", True, self.COLOR_UI_TEXT)
        self.screen.blit(shift_text, (self.width - shift_text.get_width() - 20, self.height - bottom_panel_h + 15))

        cx, cy = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        cs = 10
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cx - cs, cy), (cx + cs, cy))
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cx, cy - cs), (cx, cy + cs))

        if self.game_over:
            outcome_text = "SYSTEM STABILIZED" if self.grid_health > 0 else "SYSTEM OVERLOAD"
            color = self.COLOR_HEALTH if self.grid_health > 0 else self.COLOR_DISTORTION
            text_surf = self.font_large.render(outcome_text, True, color)
            text_rect = text_surf.get_rect(center=(self.width//2, self.height//2))
            pygame.draw.rect(self.screen, self.COLOR_BG, text_rect.inflate(20, 20))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "grid_health": self.grid_health,
            "energy": self.energy,
            "prisms": len(self.prisms),
            "distortions": len(self.distortions)
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()
        elif self.render_mode == "human":
            if self.display is None:
                self.display = pygame.display.set_mode((self.width, self.height))
            
            # The _get_observation method renders the game state to self.screen
            obs_array = self._get_observation()
            
            # Blit the screen surface to the display
            self.display.blit(self.screen, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return obs_array

    def close(self):
        if self.display is not None:
            pygame.display.quit()
            self.display = None
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)

    env.close()