import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:59:40.664022
# Source Brief: brief_00201.md
# Brief Index: 201
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    game_description = (
        "Defend your quantum shelter's core from incoming anomalies. Collect resources to expand "
        "your shelter and build defenses within its protective grid."
    )
    user_guide = (
        "Controls: Use ↑↓←→ to move within the shelter. Press space to teleport to the core "
        "(or expand it). Press shift to build a defense turret."
    )
    auto_advance = True
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 40
        self.GRID_W, self.GRID_H = self.WIDTH // self.GRID_SIZE, self.HEIGHT // self.GRID_SIZE
        self.CENTER_GRID_POS = (self.GRID_W // 2, self.GRID_H // 2)
        
        self.MAX_STEPS = 2000
        
        # --- Colors ---
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_GRID = (20, 40, 70)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 255, 50)
        self.COLOR_RESOURCE = (0, 255, 100)
        self.COLOR_RESOURCE_GLOW = (0, 255, 100, 60)
        self.COLOR_DEFENSE = (255, 200, 0)
        self.COLOR_DEFENSE_GLOW = (255, 200, 0, 70)
        self.COLOR_ANOMALY = (255, 50, 50)
        self.COLOR_ANOMALY_GLOW = (255, 50, 50, 80)
        self.COLOR_CORE = (200, 200, 255)
        self.COLOR_CORE_GLOW = (200, 200, 255, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_BAR_BG = (30, 50, 90)
        self.COLOR_BAR_FILL = (100, 180, 255)

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        self.render_mode = render_mode
        self.human_screen = None
        if self.render_mode == "human":
            self.human_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Quantum Shelter")

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        self.game_over = False
        
        self.player_grid_pos = (0, 0)
        self.player_pixel_pos = [0.0, 0.0]
        self.resources = 0
        self.max_resources = 100
        self.shelter_size = 0
        
        self.defenses = []
        self.anomalies = []
        self.resource_nodes = []
        self.particles = []
        
        self.anomaly_spawn_timer = 0
        self.base_anomaly_speed = 0.0
        
        self.last_space_held = False
        self.last_shift_held = False

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_grid_pos = self.CENTER_GRID_POS
        self.player_pixel_pos = [self.player_grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE/2, 
                                 self.player_grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE/2]

        self.resources = 10
        self.shelter_size = 1
        
        self.defenses = []
        self.anomalies = []
        self.resource_nodes = []
        self.particles = []

        for _ in range(5):
            self._spawn_resource_node()
        
        self.anomaly_spawn_timer = 60
        self.base_anomaly_speed = 0.75
        
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.reward_this_step = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_input(action)
        self._update_game_state()
        
        self.steps += 1
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = False
        
        if terminated and not self.game_over:
            # Survived full episode
            self.reward_this_step += 100
            self.score += 100
        
        if self.render_mode == "human":
            self.render()

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1  # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1  # Right
        
        if dx != 0 or dy != 0:
            new_pos = (self.player_grid_pos[0] + dx, self.player_grid_pos[1] + dy)
            if self._is_within_shelter(new_pos):
                self.player_grid_pos = new_pos
                # Sound effect placeholder: player_move.wav

        # --- Space Action (Teleport / Expand) ---
        if space_held and not self.last_space_held:
            if self.player_grid_pos == self.CENTER_GRID_POS:
                self._expand_shelter()
            else:
                self._teleport_to_base()
        
        # --- Shift Action (Build Defense) ---
        if shift_held and not self.last_shift_held:
            self._build_defense()
            
        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_game_state(self):
        # Smooth player pixel position
        target_px = [self.player_grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE/2, 
                     self.player_grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE/2]
        self.player_pixel_pos[0] += (target_px[0] - self.player_pixel_pos[0]) * 0.5
        self.player_pixel_pos[1] += (target_px[1] - self.player_pixel_pos[1]) * 0.5

        # Update anomalies
        current_anomaly_speed = self.base_anomaly_speed + 0.05 * (self.steps // 200)
        for anomaly in self.anomalies[:]:
            anomaly['pos'][0] += anomaly['vel'][0] * current_anomaly_speed
            anomaly['pos'][1] += anomaly['vel'][1] * current_anomaly_speed
            
            # Check for collision with core
            core_rect = pygame.Rect(self.CENTER_GRID_POS[0] * self.GRID_SIZE,
                                    self.CENTER_GRID_POS[1] * self.GRID_SIZE,
                                    self.GRID_SIZE, self.GRID_SIZE)
            if core_rect.collidepoint(anomaly['pos']):
                self.game_over = True
                self.reward_this_step -= 100
                self.score -= 100
                # Sound effect placeholder: core_breach.wav
                self._create_particles(self.player_pixel_pos, 100, self.COLOR_ANOMALY, 5, 20)
                break
            
            # Check for collision with defenses
            anomaly_grid_pos = (int(anomaly['pos'][0] // self.GRID_SIZE), int(anomaly['pos'][1] // self.GRID_SIZE))
            for defense in self.defenses[:]:
                if defense['pos'] == anomaly_grid_pos:
                    self.anomalies.remove(anomaly)
                    self.defenses.remove(defense)
                    self.reward_this_step += 1
                    self.score += 1
                    # Sound effect placeholder: anomaly_destroyed.wav
                    self._create_particles(anomaly['pos'], 50, self.COLOR_DEFENSE, 3, 15)
                    break
        
        # Player-resource collection
        for node in self.resource_nodes[:]:
            if node['active'] and node['pos'] == self.player_grid_pos:
                node['active'] = False
                node['timer'] = 30 # Respawn time
                self.resources = min(self.max_resources, self.resources + 10)
                self.reward_this_step += 0.1
                self.score += 0.1
                # Sound effect placeholder: resource_collect.wav
                px, py = (node['pos'][0] * self.GRID_SIZE + self.GRID_SIZE/2, 
                          node['pos'][1] * self.GRID_SIZE + self.GRID_SIZE/2)
                self._create_particles((px, py), 20, self.COLOR_RESOURCE, 2, 10)

        # Resource respawn
        for node in self.resource_nodes:
            if not node['active']:
                node['timer'] -= 1
                if node['timer'] <= 0:
                    node['active'] = True
                    # Sound effect placeholder: resource_respawn.wav

        # Anomaly spawning
        self.anomaly_spawn_timer -= 1
        if self.anomaly_spawn_timer <= 0:
            self._spawn_anomaly()
            self.anomaly_spawn_timer = max(15, 80 - self.steps // 50)

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _is_within_shelter(self, grid_pos):
        cx, cy = self.CENTER_GRID_POS
        return abs(grid_pos[0] - cx) <= self.shelter_size and abs(grid_pos[1] - cy) <= self.shelter_size

    def _expand_shelter(self):
        cost = 15 * self.shelter_size
        if self.resources >= cost:
            self.resources -= cost
            self.shelter_size += 1
            self.reward_this_step += 5
            self.score += 5
            # Sound effect placeholder: shelter_expand.wav
            px, py = (self.CENTER_GRID_POS[0] * self.GRID_SIZE + self.GRID_SIZE/2, 
                      self.CENTER_GRID_POS[1] * self.GRID_SIZE + self.GRID_SIZE/2)
            self._create_particles((px, py), 50, self.COLOR_CORE, 4, 30)
        else:
            # Sound effect placeholder: action_fail.wav
            pass

    def _teleport_to_base(self):
        cost = 1
        if self.resources >= cost:
            self.resources -= cost
            self._create_particles(self.player_pixel_pos, 30, self.COLOR_PLAYER, 3, 15)
            self.player_grid_pos = self.CENTER_GRID_POS
            # Sound effect placeholder: teleport.wav
        else:
            # Sound effect placeholder: action_fail.wav
            pass

    def _build_defense(self):
        cost = 5
        is_occupied = any(d['pos'] == self.player_grid_pos for d in self.defenses)
        is_core = self.player_grid_pos == self.CENTER_GRID_POS

        if self.resources >= cost and not is_occupied and not is_core:
            self.resources -= cost
            self.defenses.append({'pos': self.player_grid_pos})
            # Sound effect placeholder: build_defense.wav
            px, py = (self.player_grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE/2, 
                      self.player_grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE/2)
            self._create_particles((px, py), 20, self.COLOR_DEFENSE, 2, 10)
        else:
            # Sound effect placeholder: action_fail.wav
            pass
            
    def _spawn_resource_node(self):
        while True:
            pos = (self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H))
            if pos != self.CENTER_GRID_POS and not any(n['pos'] == pos for n in self.resource_nodes):
                self.resource_nodes.append({'pos': pos, 'active': True, 'timer': 0})
                break

    def _spawn_anomaly(self):
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = [self.np_random.uniform(0, self.WIDTH), -20]
        elif edge == 1: # Bottom
            pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20]
        elif edge == 2: # Left
            pos = [-20, self.np_random.uniform(0, self.HEIGHT)]
        else: # Right
            pos = [self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT)]

        target_pos = [self.CENTER_GRID_POS[0] * self.GRID_SIZE + self.GRID_SIZE/2,
                      self.CENTER_GRID_POS[1] * self.GRID_SIZE + self.GRID_SIZE/2]

        # Introduce pattern variation based on steps
        pattern_type = 'linear'
        if self.steps > 500:
            if self.np_random.random() < 0.3:
                pattern_type = 'spiral'
        
        angle = math.atan2(target_pos[1] - pos[1], target_pos[0] - pos[0])
        vel = [math.cos(angle), math.sin(angle)]

        if pattern_type == 'spiral':
            # Add a perpendicular component to velocity for spiral motion
            vel[0] += math.cos(angle + math.pi/2) * 0.5
            vel[1] += math.sin(angle + math.pi/2) * 0.5
            # Normalize
            mag = math.sqrt(vel[0]**2 + vel[1]**2)
            if mag > 0:
                vel[0] /= mag
                vel[1] /= mag

        self.anomalies.append({'pos': pos, 'vel': vel, 'type': pattern_type})
        # Sound effect placeholder: anomaly_spawn.wav

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw build grid
        cx, cy = self.CENTER_GRID_POS
        for r in range(1, self.shelter_size + 1):
            alpha = max(0, 100 - r * 20)
            color = self.COLOR_GRID[:3] + (alpha,)
            rect = pygame.Rect((cx - r) * self.GRID_SIZE, (cy - r) * self.GRID_SIZE,
                               (2 * r + 1) * self.GRID_SIZE, (2 * r + 1) * self.GRID_SIZE)
            pygame.draw.rect(self.screen, color, rect, 1)

        # Draw shelter core
        self._draw_glow(self.CENTER_GRID_POS, self.GRID_SIZE * 0.8, self.COLOR_CORE_GLOW)
        core_rect = pygame.Rect(self.CENTER_GRID_POS[0] * self.GRID_SIZE + self.GRID_SIZE*0.2,
                                self.CENTER_GRID_POS[1] * self.GRID_SIZE + self.GRID_SIZE*0.2,
                                self.GRID_SIZE*0.6, self.GRID_SIZE*0.6)
        pygame.draw.rect(self.screen, self.COLOR_CORE, core_rect, border_radius=3)
        
        # Draw resource nodes
        for node in self.resource_nodes:
            if node['active']:
                self._draw_glow(node['pos'], self.GRID_SIZE * 0.7, self.COLOR_RESOURCE_GLOW)
                px, py = (node['pos'][0] * self.GRID_SIZE + self.GRID_SIZE/2, 
                          node['pos'][1] * self.GRID_SIZE + self.GRID_SIZE/2)
                pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), int(self.GRID_SIZE*0.3), self.COLOR_RESOURCE)
                pygame.gfxdraw.aacircle(self.screen, int(px), int(py), int(self.GRID_SIZE*0.3), self.COLOR_RESOURCE)

        # Draw defenses
        for defense in self.defenses:
            self._draw_glow(defense['pos'], self.GRID_SIZE * 0.6, self.COLOR_DEFENSE_GLOW)
            rect = pygame.Rect(defense['pos'][0] * self.GRID_SIZE + self.GRID_SIZE*0.25,
                               defense['pos'][1] * self.GRID_SIZE + self.GRID_SIZE*0.25,
                               self.GRID_SIZE*0.5, self.GRID_SIZE*0.5)
            pygame.draw.rect(self.screen, self.COLOR_DEFENSE, rect, border_radius=2)

        # Draw anomalies
        for anomaly in self.anomalies:
            px, py = int(anomaly['pos'][0]), int(anomaly['pos'][1])
            size = int(self.GRID_SIZE * 0.25)
            self._draw_glow_pixel((px, py), self.GRID_SIZE * 0.5, self.COLOR_ANOMALY_GLOW)
            # Draw distorted wave
            for i in range(3):
                offset = math.sin(self.steps * 0.2 + i * math.pi/2) * 4
                pygame.gfxdraw.filled_circle(self.screen, px + int(offset), py, size, self.COLOR_ANOMALY)
                pygame.gfxdraw.aacircle(self.screen, px + int(offset), py, size, self.COLOR_ANOMALY)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = p['color'][:3] + (alpha,)
            size = int(p['size'] * (p['lifespan'] / p['max_lifespan']))
            if size > 0:
                rect = pygame.Rect(int(p['pos'][0] - size/2), int(p['pos'][1] - size/2), size, size)
                shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
                self.screen.blit(shape_surf, rect)

        # Draw player
        px, py = int(self.player_pixel_pos[0]), int(self.player_pixel_pos[1])
        self._draw_glow_pixel((px, py), self.GRID_SIZE * 0.8, self.COLOR_PLAYER_GLOW)
        size1 = int(self.GRID_SIZE * 0.35 + math.sin(self.steps * 0.1) * 3)
        size2 = int(self.GRID_SIZE * 0.35 + math.cos(self.steps * 0.1) * 3)
        p1 = (px - size1, py)
        p2 = (px, py - size2)
        p3 = (px + size1, py)
        p4 = (px, py + size2)
        pygame.gfxdraw.aapolygon(self.screen, [p1,p2,p3,p4], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1,p2,p3,p4], self.COLOR_PLAYER)

    def _render_ui(self):
        # Shelter Size
        size_text = self.font_small.render(f"SHELTER SIZE: {self.shelter_size}", True, self.COLOR_TEXT)
        self.screen.blit(size_text, (10, 10))

        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 30))

        # Resources Bar
        bar_w, bar_h = 150, 20
        bar_x, bar_y = self.WIDTH - bar_w - 10, 10
        fill_w = int((self.resources / self.max_resources) * bar_w)
        
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        if fill_w > 0:
            pygame.draw.rect(self.screen, self.COLOR_BAR_FILL, (bar_x, bar_y, fill_w, bar_h), border_radius=3)
        resource_text = self.font_small.render("RESOURCES", True, self.COLOR_TEXT)
        self.screen.blit(resource_text, (bar_x + (bar_w - resource_text.get_width()) // 2, bar_y + bar_h))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,150))
            self.screen.blit(overlay, (0,0))
            game_over_text = self.font_large.render("CORE BREACHED", True, self.COLOR_ANOMALY)
            text_rect = game_over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "shelter_size": self.shelter_size,
            "resources": self.resources,
        }
        
    def _draw_glow(self, grid_pos, radius, color):
        px, py = (grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE/2, 
                  grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE/2)
        self._draw_glow_pixel((px, py), radius, color)

    def _draw_glow_pixel(self, pixel_pos, radius, color):
        px, py = int(pixel_pos[0]), int(pixel_pos[1])
        rad = int(radius)
        glow_surf = pygame.Surface((rad * 2, rad * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, color, (rad, rad), rad)
        self.screen.blit(glow_surf, (px - rad, py - rad), special_flags=pygame.BLEND_RGBA_ADD)

    def _create_particles(self, pos, count, color, speed, lifespan_max):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_mag = self.np_random.uniform(0.5, 1.0) * speed
            vel = [math.cos(angle) * vel_mag, math.sin(angle) * vel_mag]
            lifespan = self.np_random.integers(lifespan_max // 2, lifespan_max)
            self.particles.append({
                'pos': list(pos), 
                'vel': vel, 
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()
        elif self.render_mode == "human":
            if self.human_screen is None:
                pygame.init()
                self.human_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            
            # The observation is already the rendered screen
            obs_array = self._get_observation()
            # The observation is HxWxC, but pygame wants WxHxC. Need to transpose back.
            surf = pygame.surfarray.make_surface(np.transpose(obs_array, (1, 0, 2)))
            self.human_screen.blit(surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

    def close(self):
        if self.human_screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.human_screen = None

if __name__ == "__main__":
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    done = False
    total_reward = 0
    
    # --- Manual Play Controls ---
    # Arrows: Move
    # Space: Teleport to base / Expand shelter
    # Shift: Build defense
    # Q: Quit
    
    print("Controls: Arrow keys to move, Space to teleport/expand, Left Shift to build defense, Q to quit.")
    
    while not done:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                done = True

        if done:
            break

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Give time to see the game over screen
            pygame.time.wait(2000)
            done = True

    env.close()