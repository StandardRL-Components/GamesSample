import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Collect colored resources to match the energy sequence before the storm consumes you. "
        "Survive as many storm cycles as you can."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to collect nearby resources and "
        "shift to match a resource with an energy source."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG_DEEP = (10, 5, 20)
    COLOR_BG_LIGHT = (30, 15, 60)
    COLOR_PLAYER = (220, 220, 255)
    COLOR_PLAYER_GLOW = (150, 150, 255)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BAR_BG = (50, 50, 80)
    COLOR_UI_ENERGY = (0, 255, 255)
    COLOR_UI_ENERGY_LOW = (255, 100, 0)
    
    # FIX: The collectible resources must match the energy source types ("red", "green", "blue").
    RESOURCE_COLORS = {
        "red": (255, 80, 80),
        "green": (80, 255, 80),
        "blue": (80, 80, 255),
    }
    ENERGY_SOURCE_COLORS = {
        "red": (255, 50, 50),
        "green": (50, 255, 50),
        "blue": (50, 50, 255)
    }
    STORM_COLOR = (200, 200, 220)
    
    # Game Parameters
    POD_MAX_SPEED = 5.0
    POD_ACCELERATION = 0.8
    POD_FRICTION = 0.90
    POD_SIZE = 12
    COLLECTION_RADIUS = 30
    
    STORM_CYCLE_LENGTH = 900 # 30 seconds at 30 FPS
    STORM_DURATION = 300 # 10 seconds at 30 FPS
    INITIAL_STORM_DRAIN = 0.5
    STORM_DRAIN_INCREMENT = 0.2
    
    MAX_RESOURCES = 10
    WIN_CYCLE = 10
    MAX_STEPS = 1000 * 5 # 5 minutes max episode length

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # State variables are initialized in reset()
        self.pod_pos = None
        self.pod_velocity = None
        self.pod_energy = None
        self.resources = None
        self.energy_sources = None
        self.energy_match_sequence = None
        self.current_match_index = None
        self.storm_cycle = None
        self.storm_timer = None
        self.storm_intensity = None
        self.is_storming = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.particles = None
        self.nebula_stars = None
        self.low_energy_penalty_given = None
        self.previous_space_held = False
        self.previous_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.pod_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.pod_velocity = pygame.Vector2(0, 0)
        self.pod_energy = 100.0
        
        self.resources = []
        self.energy_sources = []
        
        self.energy_match_sequence = ["red", "green", "blue"]
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.energy_match_sequence)
        self.current_match_index = 0
        
        self.storm_cycle = 1
        self.storm_timer = self.STORM_CYCLE_LENGTH
        self.storm_intensity = self.INITIAL_STORM_DRAIN
        self.is_storming = False
        
        self.player_resources = {color: 0 for color in self.RESOURCE_COLORS.keys()}
        
        self.particles = []
        self.low_energy_penalty_given = False
        self.previous_space_held = False
        self.previous_shift_held = False

        self._spawn_nebula()
        self._spawn_energy_sources()
        for _ in range(self.MAX_RESOURCES):
            self._spawn_resource()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement, space_action, shift_action = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.0
        self.steps += 1
        
        self._handle_movement(movement)
        
        space_pressed = space_action and not self.previous_space_held
        if space_pressed:
            reward += self._collect_resources()

        shift_pressed = shift_action and not self.previous_shift_held
        if shift_pressed:
            match_reward, matched_all = self._match_energy()
            reward += match_reward
            if matched_all:
                self.pod_energy = 100.0
                self._spawn_energy_sources()
        
        self.previous_space_held = space_action
        self.previous_shift_held = shift_action

        self._update_pod()
        self._update_storm()
        self._update_particles()
        
        if len(self.resources) < self.MAX_RESOURCES:
            self._spawn_resource()
            
        if self.pod_energy < 25 and not self.low_energy_penalty_given:
            reward -= 5.0
            self.low_energy_penalty_given = True
        elif self.pod_energy >= 25:
            self.low_energy_penalty_given = False
            
        self.score += reward
        
        terminated = False
        if self.pod_energy <= 0:
            reward -= 100.0
            terminated = True
            self.game_over = True
        
        if self.storm_cycle > self.WIN_CYCLE:
            reward += 100.0
            terminated = True
            self.game_over = True
        
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        if terminated or truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_movement(self, movement_action):
        acceleration_vec = pygame.Vector2(0, 0)
        if movement_action == 1: # Up
            acceleration_vec.y = -self.POD_ACCELERATION
        elif movement_action == 2: # Down
            acceleration_vec.y = self.POD_ACCELERATION
        elif movement_action == 3: # Left
            acceleration_vec.x = -self.POD_ACCELERATION
        elif movement_action == 4: # Right
            acceleration_vec.x = self.POD_ACCELERATION
        
        self.pod_velocity += acceleration_vec
        if self.pod_velocity.length() > self.POD_MAX_SPEED:
            self.pod_velocity.scale_to_length(self.POD_MAX_SPEED)

    def _update_pod(self):
        self.pod_velocity *= self.POD_FRICTION
        self.pod_pos += self.pod_velocity
        
        self.pod_pos.x = max(self.POD_SIZE, min(self.pod_pos.x, self.SCREEN_WIDTH - self.POD_SIZE))
        self.pod_pos.y = max(self.POD_SIZE, min(self.pod_pos.y, self.SCREEN_HEIGHT - self.POD_SIZE))

    def _update_storm(self):
        self.storm_timer -= 1
        if self.storm_timer <= 0:
            if self.is_storming:
                self.is_storming = False
                self.storm_timer = self.STORM_CYCLE_LENGTH
                self.storm_cycle += 1
                self.storm_intensity += self.STORM_DRAIN_INCREMENT
            else:
                self.is_storming = True
                self.storm_timer = self.STORM_DURATION
        
        if self.is_storming:
            self.pod_energy -= self.storm_intensity / self.FPS
            self.pod_energy = max(0, self.pod_energy)

    def _collect_resources(self):
        collected_reward = 0.0
        for res in self.resources[:]:
            dist = self.pod_pos.distance_to(res['pos'])
            if dist < self.COLLECTION_RADIUS:
                self.player_resources[res['type']] += 1
                self._create_particles(res['pos'], self.RESOURCE_COLORS[res['type']], 10)
                self.resources.remove(res)
                collected_reward += 0.1
        return collected_reward

    def _match_energy(self):
        if self.current_match_index >= len(self.energy_match_sequence):
            return 0.0, False

        required_color = self.energy_match_sequence[self.current_match_index]
        if self.player_resources[required_color] > 0:
            for source in self.energy_sources:
                if not source['matched'] and source['type'] == required_color:
                    dist = self.pod_pos.distance_to(source['pos'])
                    if dist < self.COLLECTION_RADIUS + 10:
                        self.player_resources[required_color] -= 1
                        source['matched'] = True
                        self.current_match_index += 1
                        self._create_particles(source['pos'], self.ENERGY_SOURCE_COLORS[source['type']], 30, 3)
                        
                        is_set_complete = self.current_match_index == len(self.energy_match_sequence)
                        reward = 5.0 if is_set_complete else 1.0
                        return reward, is_set_complete
        return 0.0, False

    def _spawn_resource(self):
        res_type = random.choice(list(self.RESOURCE_COLORS.keys()))
        pos = pygame.Vector2(random.randint(20, self.SCREEN_WIDTH - 20),
                             random.randint(20, self.SCREEN_HEIGHT - 20))
        self.resources.append({'type': res_type, 'pos': pos, 'anim_offset': random.uniform(0, math.pi * 2)})

    def _spawn_energy_sources(self):
        self.energy_sources.clear()
        self.current_match_index = 0
        random.shuffle(self.energy_match_sequence)
        
        positions = []
        for i, color_type in enumerate(self.energy_match_sequence):
            while True:
                pos = pygame.Vector2(random.randint(50, self.SCREEN_WIDTH - 50),
                                     random.randint(50, self.SCREEN_HEIGHT - 50))
                too_close = False
                for existing_pos in positions:
                    if pos.distance_to(existing_pos) < 100:
                        too_close = True
                        break
                if not too_close:
                    positions.append(pos)
                    break
            
            self.energy_sources.append({
                'type': color_type,
                'pos': pos,
                'matched': False,
                'anim_offset': random.uniform(0, math.pi * 2)
            })
            
    def _spawn_nebula(self):
        self.nebula_stars = []
        for _ in range(150):
            self.nebula_stars.append({
                'pos': pygame.Vector2(random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)),
                'size': random.uniform(0.5, 2.0),
                'speed': random.uniform(0.1, 0.3),
                'color': random.choice([self.COLOR_BG_DEEP, self.COLOR_BG_LIGHT, (20, 10, 40)])
            })

    def _create_particles(self, pos, color, count, speed_mult=1):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'color': color, 'life': random.randint(10, 20)})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "pod_energy": self.pod_energy,
            "storm_cycle": self.storm_cycle,
            "resources": self.player_resources.copy()
        }

    def _render_background(self):
        self.screen.fill(self.COLOR_BG_DEEP)
        for star in self.nebula_stars:
            star['pos'].x = (star['pos'].x + star['speed']) % self.SCREEN_WIDTH
            pygame.draw.circle(self.screen, star['color'], star['pos'], star['size'])
            
    def _render_game(self):
        for source in self.energy_sources:
            pos = source['pos']
            color = self.ENERGY_SOURCE_COLORS[source['type']]
            base_radius = 20
            anim_scale = 1 + 0.1 * math.sin(pygame.time.get_ticks() / 300 + source['anim_offset'])
            
            if source['matched']:
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(base_radius * 0.8), (*color, 50))
                pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(base_radius * 0.8), (*color, 100))
            else:
                for i in range(5):
                    alpha = 150 - i * 30
                    radius = int(base_radius * anim_scale + i * 2)
                    pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), radius, (*color, alpha))
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(base_radius * 0.5), color)

        for res in self.resources:
            pos = res['pos']
            color = self.RESOURCE_COLORS[res['type']]
            size = 5 + 1.5 * math.sin(pygame.time.get_ticks() / 200 + res['anim_offset'])
            pygame.draw.circle(self.screen, color, pos, size)
            
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color_with_alpha = (*p['color'], alpha)
            pygame.draw.circle(self.screen, color_with_alpha, p['pos'], max(0, p['life'] * 0.2))

        pos = self.pod_pos
        for i in range(4):
            glow_alpha = 80 - i * 20
            glow_radius = self.POD_SIZE + i * 3
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), glow_radius, (*self.COLOR_PLAYER_GLOW, glow_alpha))
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.POD_SIZE, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), self.POD_SIZE, self.COLOR_PLAYER)
        
        if self.pod_velocity.length() > 0.5:
            for i in range(3):
                p_pos = pos - self.pod_velocity.normalize() * (self.POD_SIZE + i * 2)
                p_size = max(1, self.POD_SIZE/2 - i)
                pygame.draw.circle(self.screen, self.COLOR_UI_ENERGY, p_pos, p_size)

    def _render_ui(self):
        bar_width = 200
        bar_height = 15
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = 10
        energy_percent = self.pod_energy / 100.0
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        energy_color = self.COLOR_UI_ENERGY if energy_percent > 0.25 else self.COLOR_UI_ENERGY_LOW
        pygame.draw.rect(self.screen, energy_color, (bar_x, bar_y, bar_width * energy_percent, bar_height), border_radius=3)
        
        cycle_text = self.font_small.render(f"CYCLE: {self.storm_cycle}/{self.WIN_CYCLE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(cycle_text, (self.SCREEN_WIDTH - cycle_text.get_width() - 10, 10))
        
        if self.is_storming:
            storm_text = self.font_large.render("STORM", True, self.COLOR_UI_ENERGY_LOW)
            text_rect = storm_text.get_rect(center=(self.SCREEN_WIDTH/2, 40))
            self.screen.blit(storm_text, text_rect)
            if self.steps % 4 < 2:
                storm_overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                storm_overlay.fill((*self.STORM_COLOR, 30))
                self.screen.blit(storm_overlay, (0,0))
        
        start_x = 10
        start_y = self.SCREEN_HEIGHT - 30
        for i, (color_name, count) in enumerate(self.player_resources.items()):
            color_val = self.RESOURCE_COLORS[color_name]
            pygame.draw.circle(self.screen, color_val, (start_x + i * 60, start_y), 10)
            count_text = self.font_large.render(f"{count}", True, self.COLOR_UI_TEXT)
            self.screen.blit(count_text, (start_x + 15 + i * 60, start_y - 12))
            
        if self.current_match_index < len(self.energy_match_sequence):
            required_color_name = self.energy_match_sequence[self.current_match_index]
            required_color_val = self.ENERGY_SOURCE_COLORS[required_color_name]
            
            match_text = self.font_small.render("MATCH:", True, self.COLOR_UI_TEXT)
            self.screen.blit(match_text, (self.SCREEN_WIDTH / 2 - 50, self.SCREEN_HEIGHT - 25))
            pygame.draw.circle(self.screen, required_color_val, (self.SCREEN_WIDTH / 2 + 15, self.SCREEN_HEIGHT - 18), 8)
            
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Nebula Survival Pod")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset(seed=random.randint(0, 10000))
            total_reward = 0
            
        clock.tick(GameEnv.FPS)
        
    env.close()