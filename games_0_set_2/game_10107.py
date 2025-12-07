import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player gathers resources and places enchanted
    trinkets to ward off ghostly creatures, while building a hideout in a haunted forest.
    The goal is to reach a hideout strength of 1000.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = "Gather resources to build a hideout and place enchanted trinkets to ward off ghostly creatures. Reach a hideout strength of 1000 to win."
    user_guide = "Use the arrow keys (↑↓←→) to move the cursor. Press space to place a trinket or upgrade the selected skill. Press shift to cycle between skills."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.render_mode = render_mode
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game Constants ---
        self.MAX_STEPS = 5000
        self.GATHERING_ZONE_POS = pygame.Vector2(self.width / 2, self.height / 2 + 50)
        self.GATHERING_ZONE_RADIUS = 40
        self.VICTORY_STRENGTH = 1000
        self.LOSS_OVERRUN_SECONDS = 10

        # --- Colors ---
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_GHOST = (255, 80, 80)
        self.COLOR_TRINKET = (100, 150, 255)
        self.COLOR_RESOURCE = (80, 255, 80)
        self.COLOR_HIDEOUT = (180, 100, 255)
        self.COLOR_UI_TEXT = (230, 230, 240)
        self.COLOR_CURSOR = (255, 255, 0)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.game_over = False
        self.resources = 0
        self.hideout_strength = 0
        self.cursor_pos = pygame.Vector2(0, 0)
        self.trinkets = []
        self.ghosts = []
        self.resource_nodes = []
        self.particles = []
        self.skills = []
        self.selected_skill_index = 0
        self.ghost_spawn_timer = 0
        self.ghost_spawn_rate = 300 # 10 seconds at 30fps
        self.max_ghosts_at_once = 1
        self.base_ghost_speed = 0.5
        self.resource_spawn_timer = 0
        self.overrun_steps = 0
        self.last_space_held = False
        self.last_shift_held = False


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.game_over = False
        self.resources = 50  # Start with enough for a trinket
        self.hideout_strength = 0
        
        self.cursor_pos = pygame.Vector2(self.width / 2, self.height / 2)
        self.trinkets = []
        self.ghosts = []
        self.resource_nodes = []
        self.particles = []

        # --- Skill Tree Definition ---
        self.skills = [
            {'name': 'Build Hideout', 'cost': 10, 'level': 0, 'max_level': 999, 'desc': "+10 Strength"},
            {'name': 'Gather Speed', 'cost': 50, 'level': 0, 'max_level': 5, 'desc': "+20% Speed"},
            {'name': 'Trinket Power', 'cost': 75, 'level': 0, 'max_level': 5, 'desc': "+15% Repel"},
        ]
        self.selected_skill_index = 0
        
        # --- Difficulty Scaling ---
        self.ghost_spawn_timer = 0
        self.ghost_spawn_rate = 300 # 10 seconds at 30fps
        self.max_ghosts_at_once = 1
        self.base_ghost_speed = 0.5
        self.resource_spawn_timer = 0
        
        # --- Loss Condition ---
        self.overrun_steps = 0
        
        # --- Input State ---
        self.last_space_held = False
        self.last_shift_held = False
        
        # --- Initial State ---
        for _ in range(5):
            self._spawn_resource_node()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, True, self._get_info()

        self.steps += 1
        reward = 0.0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        
        # --- Handle Player Actions ---
        self._handle_movement(movement)
        
        if shift_press:
            self.selected_skill_index = (self.selected_skill_index + 1) % len(self.skills)
            
        if space_press:
            # Check if cursor is over the skill UI
            ui_rect = pygame.Rect(0, 0, self.width, 50)
            if ui_rect.collidepoint(self.cursor_pos):
                reward += self._upgrade_skill()
            else:
                reward += self._place_trinket()

        # --- Update Game Logic ---
        self._update_difficulty()
        self._spawn_entities()
        self._update_ghosts()
        reward += self._update_gathering()
        self._update_particles()
        
        # --- Termination and Terminal Rewards ---
        terminated = False
        truncated = False
        if self.hideout_strength >= self.VICTORY_STRENGTH:
            reward += 100
            terminated = True
        elif self.overrun_steps / self.metadata['render_fps'] >= self.LOSS_OVERRUN_SECONDS:
            reward -= 100
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True

        self.game_over = terminated or truncated

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_movement(self, movement):
        cursor_speed = 8
        if movement == 1: self.cursor_pos.y -= cursor_speed
        elif movement == 2: self.cursor_pos.y += cursor_speed
        elif movement == 3: self.cursor_pos.x -= cursor_speed
        elif movement == 4: self.cursor_pos.x += cursor_speed
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.width)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.height)

    def _upgrade_skill(self):
        skill = self.skills[self.selected_skill_index]
        if skill['level'] < skill['max_level'] and self.resources >= skill['cost']:
            self.resources -= skill['cost']
            skill['level'] += 1
            if skill['name'] == 'Build Hideout':
                self.hideout_strength += 10
                skill['cost'] = int(skill['cost'] * 1.05) # Build cost increases slowly
            else:
                skill['cost'] = int(skill['cost'] * 2)
            
            self._create_particles(self.GATHERING_ZONE_POS, 30, self.COLOR_HIDEOUT, 2)
            return 5.0
        return 0.0

    def _place_trinket(self):
        cost = 25
        if self.resources >= cost:
            self.resources -= cost
            self.trinkets.append({'pos': self.cursor_pos.copy(), 'strength': 1.0})
            self._create_particles(self.cursor_pos, 20, self.COLOR_TRINKET, 1.5)
            return 1.0
        return 0.0

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.base_ghost_speed += 0.01
        if self.steps > 0 and self.steps % 500 == 0:
            self.max_ghosts_at_once += 1
            self.ghost_spawn_rate = max(60, self.ghost_spawn_rate - 15)

    def _spawn_entities(self):
        self.ghost_spawn_timer += 1
        if self.ghost_spawn_timer >= self.ghost_spawn_rate and len(self.ghosts) < self.max_ghosts_at_once:
            self.ghost_spawn_timer = 0
            edge = self.np_random.integers(0, 4)
            if edge == 0: pos = pygame.Vector2(self.np_random.uniform(0, self.width), -10)
            elif edge == 1: pos = pygame.Vector2(self.np_random.uniform(0, self.width), self.height + 10)
            elif edge == 2: pos = pygame.Vector2(-10, self.np_random.uniform(0, self.height))
            else: pos = pygame.Vector2(self.width + 10, self.np_random.uniform(0, self.height))
            self.ghosts.append({'pos': pos, 'vel': pygame.Vector2(0,0)})
        
        self.resource_spawn_timer += 1
        if self.resource_spawn_timer >= 120 and len(self.resource_nodes) < 15: # 4 seconds
            self.resource_spawn_timer = 0
            self._spawn_resource_node()
            
    def _spawn_resource_node(self):
        pos = pygame.Vector2(
            self.np_random.uniform(20, self.width - 20),
            self.np_random.uniform(70, self.height - 20)
        )
        if pos.distance_to(self.GATHERING_ZONE_POS) > self.GATHERING_ZONE_RADIUS + 20:
            self.resource_nodes.append({'pos': pos, 'value': self.np_random.integers(5, 16)})

    def _update_ghosts(self):
        trinket_power = 1.0 + 0.15 * self.skills[2]['level']
        
        for ghost in self.ghosts:
            attraction_force = (self.GATHERING_ZONE_POS - ghost['pos']).normalize() * 0.1
            
            repulsion_force = pygame.Vector2(0, 0)
            for trinket in self.trinkets:
                dist_vec = ghost['pos'] - trinket['pos']
                dist = dist_vec.length()
                if 0 < dist < 100 * trinket_power:
                    strength = (1.0 - (dist / (100 * trinket_power))) * 2.5
                    repulsion_force += dist_vec.normalize() * strength
            
            ghost['vel'] += attraction_force + repulsion_force
            if ghost['vel'].length() > self.base_ghost_speed:
                ghost['vel'].scale_to_length(self.base_ghost_speed)
            ghost['pos'] += ghost['vel']

    def _update_gathering(self):
        reward = 0
        is_disrupted = any(ghost['pos'].distance_to(self.GATHERING_ZONE_POS) < self.GATHERING_ZONE_RADIUS for ghost in self.ghosts)
        
        if is_disrupted:
            self.overrun_steps += 1
            reward -= 0.1 / self.metadata['render_fps']
        else:
            self.overrun_steps = 0
            gather_speed = 1.0 + 0.2 * self.skills[1]['level']
            
            nodes_to_remove = []
            for i, node in enumerate(self.resource_nodes):
                if node['pos'].distance_to(self.GATHERING_ZONE_POS) < self.GATHERING_ZONE_RADIUS * 2:
                    gathered_amount = 0.1 * gather_speed
                    node['value'] -= gathered_amount
                    self.resources += gathered_amount
                    reward += 0.1 * gathered_amount
                    
                    if self.np_random.random() < 0.1:
                        self._create_particles(node['pos'], 1, self.COLOR_RESOURCE, 0.5, self.GATHERING_ZONE_POS)
                    
                    if node['value'] <= 0:
                        nodes_to_remove.append(i)
                        
            for i in sorted(nodes_to_remove, reverse=True):
                del self.resource_nodes[i]
                
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_effects()
        self._render_resource_nodes()
        self._render_gathering_zone()
        self._render_trinkets()
        self._render_ghosts()
        self._render_particles()
        self._render_hideout()
        self._render_cursor()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "resources": self.resources,
            "hideout_strength": self.hideout_strength,
            "steps": self.steps,
            "overrun_seconds": self.overrun_steps / self.metadata['render_fps'],
            "num_trinkets": len(self.trinkets),
            "num_ghosts": len(self.ghosts),
        }

    def _draw_glow_circle(self, surface, color, center, radius, glow_factor=2):
        for i in range(int(radius), 0, -2):
            alpha = 255 - (i / radius) * 255
            alpha = max(0, min(255, int(alpha / glow_factor)))
            pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), i, (*color, alpha))
            
    def _render_background_effects(self):
        for i in range(15):
            x = (hash(i * 10) % self.width)
            y = (hash(str(i)) % (self.height - 100)) + 80
            h = (hash(i*i) % 30) + 20
            w = h / 4
            color = (30, 20, 50)
            pygame.draw.polygon(self.screen, color, [(x, y), (x-w, y+h), (x+w, y+h)])

    def _render_resource_nodes(self):
        for node in self.resource_nodes:
            self._draw_glow_circle(self.screen, self.COLOR_RESOURCE, node['pos'], 5, glow_factor=3)

    def _render_gathering_zone(self):
        is_disrupted = any(g['pos'].distance_to(self.GATHERING_ZONE_POS) < self.GATHERING_ZONE_RADIUS for g in self.ghosts)
        color = self.COLOR_GHOST if is_disrupted else self.COLOR_HIDEOUT
        
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 * 5
        radius = self.GATHERING_ZONE_RADIUS + pulse
        
        temp_surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        self._draw_glow_circle(temp_surface, color, (radius, radius), int(radius), glow_factor=4)
        self.screen.blit(temp_surface, self.GATHERING_ZONE_POS - pygame.Vector2(radius, radius))

    def _render_trinkets(self):
        trinket_power = 1.0 + 0.15 * self.skills[2]['level']
        for trinket in self.trinkets:
            radius = int(100 * trinket_power)
            temp_surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surface, radius, radius, radius, (*self.COLOR_TRINKET, 10))
            pygame.gfxdraw.aacircle(temp_surface, radius, radius, radius, (*self.COLOR_TRINKET, 50))
            self.screen.blit(temp_surface, trinket['pos'] - pygame.Vector2(radius, radius))
            
            self._draw_glow_circle(self.screen, self.COLOR_TRINKET, trinket['pos'], 7, glow_factor=2)

    def _render_ghosts(self):
        trinket_power = 1.0 + 0.15 * self.skills[2]['level']
        for ghost in self.ghosts:
            min_dist = float('inf')
            for trinket in self.trinkets:
                min_dist = min(min_dist, ghost['pos'].distance_to(trinket['pos']))
            
            alpha = 255
            if min_dist < 100 * trinket_power:
                alpha = int(255 * (min_dist / (100 * trinket_power)))
            alpha = np.clip(alpha, 30, 255)
            
            self._draw_glow_circle(self.screen, self.COLOR_GHOST, ghost['pos'], 8, glow_factor=3)

    def _render_hideout(self):
        strength_ratio = min(1.0, self.hideout_strength / self.VICTORY_STRENGTH)
        
        base_h = 20 + 80 * strength_ratio
        base_w = 50 + 250 * strength_ratio
        base_rect = pygame.Rect(0, 0, base_w, base_h)
        base_rect.center = (self.width / 2, self.height - 40)
        pygame.draw.rect(self.screen, self.COLOR_HIDEOUT, base_rect, border_radius=5)
        
        if strength_ratio > 0.25:
            pygame.draw.rect(self.screen, self.COLOR_BG, base_rect.inflate(-15, -15), border_radius=3)
        if strength_ratio > 0.6:
            pygame.draw.circle(self.screen, self.COLOR_CURSOR, base_rect.center, 5)

    def _render_cursor(self):
        x, y = int(self.cursor_pos.x), int(self.cursor_pos.y)
        color = self.COLOR_CURSOR
        pygame.draw.line(self.screen, color, (x - 10, y), (x + 10, y), 2)
        pygame.draw.line(self.screen, color, (x, y - 10), (x, y + 10), 2)
        
    def _render_ui(self):
        ui_bg = pygame.Surface((self.width, 50), pygame.SRCALPHA)
        ui_bg.fill((20, 20, 40, 180))
        self.screen.blit(ui_bg, (0, 0))
        
        res_text = self.font_medium.render(f"RES: {int(self.resources)}", True, self.COLOR_RESOURCE)
        self.screen.blit(res_text, (10, 10))
        
        str_text = self.font_medium.render(f"STR: {self.hideout_strength}/{self.VICTORY_STRENGTH}", True, self.COLOR_HIDEOUT)
        self.screen.blit(str_text, (180, 10))
        
        skill = self.skills[self.selected_skill_index]
        skill_text = f"[{skill['name']} L{skill['level']}] Cost: {skill['cost']} ({skill['desc']})"
        skill_render = self.font_small.render(skill_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(skill_render, (10, 35))
        
        overrun_ratio = (self.overrun_steps / self.metadata['render_fps']) / self.LOSS_OVERRUN_SECONDS
        if overrun_ratio > 0:
            bar_width = self.width * overrun_ratio
            pygame.draw.rect(self.screen, self.COLOR_GHOST, (0, self.height - 5, bar_width, 5))

    def _render_game_over(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        msg = "VICTORY" if self.hideout_strength >= self.VICTORY_STRENGTH else "GAME OVER"
        color = self.COLOR_TRINKET if self.hideout_strength >= self.VICTORY_STRENGTH else self.COLOR_GHOST
            
        text = self.font_large.render(msg, True, color)
        text_rect = text.get_rect(center=(self.width / 2, self.height / 2))
        self.screen.blit(text, text_rect)

    def _create_particles(self, pos, count, color, speed_mult, target_pos=None):
        for _ in range(count):
            if target_pos:
                vel = (target_pos - pos).normalize() * self.np_random.uniform(0.5, 1.5) * speed_mult
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.np_random.uniform(0.5, 2.5) * speed_mult
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(20, 41),
                'color': color
            })
            
    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.pop(i)
                
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 40))
            color = (*p['color'], alpha)
            radius = int(p['lifespan'] / 10) + 1
            pygame.draw.circle(self.screen, color, p['pos'], radius)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block is for manual testing and visualization.
    # It is not part of the required Gymnasium interface.
    # To run, you might need to `pip install pygame`.
    
    # Un-comment the line below to run with a visible window
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    try:
        screen = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption("Haunted Hideaway")
        is_display_available = True
    except pygame.error:
        print("Pygame display unavailable. Running headlessly.")
        is_display_available = False

    obs, info = env.reset()
    done = False
    
    while not done:
        action = [0, 0, 0] # Default no-op action
        
        # Event handling for closing the window and keyboard input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        if is_display_available:
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = keys[pygame.K_SPACE]
            shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            
            action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if is_display_available:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        env.clock.tick(env.metadata['render_fps'])
        
    env.close()