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
import random

class GameEnv(gym.Env):
    """
    Mycorrhizal Madness: A stealth/strategy game where you expand a fungal network.

    The goal is to eliminate competing fungi by launching spores. The primary
    mechanic is flipping gravity to control your network's growth direction.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Expand your fungal network and eliminate competitors by launching spores. "
        "Flip gravity to control your growth and outmaneuver rival fungi."
    )
    user_guide = (
        "Use ↑/↓ to flip gravity, ←/→ to switch targets, and space to fire a spore."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2000
        self.FPS = 30

        # --- Colors ---
        self.COLOR_BG = (28, 22, 18)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_COMP_1 = (255, 60, 60)
        self.COLOR_COMP_1_GLOW = (255, 60, 60, 50)
        self.COLOR_COMP_2 = (60, 160, 255)
        self.COLOR_COMP_2_GLOW = (60, 160, 255, 50)
        self.COLOR_SPORE = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TARGET = (255, 255, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        self.soil_texture = self._create_soil_texture()

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gravity = (0, 0)
        self.player_nodes = []
        self.player_segments = []
        self.player_growth_cooldown = 0
        self.player_core_radius = 0
        self.competitors = []
        self.competitor_growth_speed_modifier = 0.0
        self.spore_count = 0
        self.projectiles = []
        self.particles = []
        self.targeted_competitor_idx = 0
        self.last_action = np.array([0, 0, 0])

        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player
        self.player_nodes = [[self.WIDTH / 2, self.HEIGHT - 25]]
        self.player_segments = []
        self.player_growth_cooldown = 0
        self.player_core_radius = 10
        self.gravity = (0, -1)  # Start growing up

        # Competitors
        self.competitors = [
            self._create_competitor(self.WIDTH * 0.2, self.HEIGHT / 2, self.COLOR_COMP_1, self.COLOR_COMP_1_GLOW),
            self._create_competitor(self.WIDTH * 0.8, self.HEIGHT / 2, self.COLOR_COMP_2, self.COLOR_COMP_2_GLOW),
        ]
        self.competitor_growth_speed_modifier = 1.0

        # Resources & Objects
        self.spore_count = 15
        self.projectiles = []
        self.particles = []

        # Controls
        self.targeted_competitor_idx = 0
        self.last_action = np.array([0, 0, 0])

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        truncated = False

        if self.game_over:
            return self._get_observation(), 0, True, truncated, self._get_info()

        self._handle_input(action)

        # --- Update Game State ---
        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.competitor_growth_speed_modifier += 0.05

        # Grow networks
        reward += self._update_player_growth()
        reward += self._update_competitor_growth()

        # Update dynamic objects
        reward += self._update_projectiles()
        self._update_particles()

        # --- Check End Conditions ---
        if self._check_player_collision():
            self.game_over = True
            reward -= 100
            self._create_explosion(self.player_nodes[0], self.COLOR_PLAYER, 100)

        if all(not c['alive'] for c in self.competitors):
            self.game_over = True
            reward += 100

        if self.steps >= self.MAX_STEPS:
            truncated = True

        terminated = self.game_over
        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.blit(self.soil_texture, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- State Update Helpers ---

    def _handle_input(self, action):
        movement, space, shift = action
        space_pressed = space == 1 and self.last_action[1] == 0

        # Gravity flip (up/down)
        if movement == 1: self.gravity = (0, -1)
        elif movement == 2: self.gravity = (0, 1)

        # Target switch (left/right)
        if movement == 3: self.targeted_competitor_idx = 0
        elif movement == 4: self.targeted_competitor_idx = 1
        
        # Ensure target is valid if a competitor is dead
        if self.competitors and not self.competitors[self.targeted_competitor_idx]['alive']:
            self.targeted_competitor_idx = 1 - self.targeted_competitor_idx

        # Launch spore
        if space_pressed and self.spore_count > 0 and self.competitors[self.targeted_competitor_idx]['alive']:
            self.spore_count -= 1
            self._launch_spore()

        self.last_action = action

    def _update_player_growth(self):
        self.player_growth_cooldown -= 1
        if self.player_growth_cooldown <= 0:
            self.player_growth_cooldown = 10  # Grow every 10 steps
            
            # Find a frontier node to grow from
            frontier_nodes = [i for i, node in enumerate(self.player_nodes) if sum(seg.count(i) for seg in self.player_segments) <= 1]
            if not frontier_nodes: frontier_nodes = list(range(len(self.player_nodes)))
            
            start_node_idx = self.np_random.choice(frontier_nodes)
            start_pos = self.player_nodes[start_node_idx]
            
            angle = math.atan2(self.gravity[1], self.gravity[0]) + self.np_random.uniform(-0.5, 0.5)
            length = self.np_random.uniform(15, 25)
            end_pos = [start_pos[0] + math.cos(angle) * length, start_pos[1] + math.sin(angle) * length]
            
            # Boundary check
            end_pos[0] = np.clip(end_pos[0], 10, self.WIDTH - 10)
            end_pos[1] = np.clip(end_pos[1], 10, self.HEIGHT - 10)

            self.player_nodes.append(end_pos)
            self.player_segments.append((start_node_idx, len(self.player_nodes) - 1))
            return 0.1 # Reward for growth
        return 0

    def _update_competitor_growth(self):
        reward = 0
        for comp in self.competitors:
            if not comp['alive']: continue
            comp['growth_cooldown'] -= 1 * self.competitor_growth_speed_modifier
            if comp['growth_cooldown'] <= 0:
                comp['growth_cooldown'] = 15
                
                start_node_idx = self.np_random.integers(len(comp['nodes']))
                start_pos = comp['nodes'][start_node_idx]
                
                angle = self.np_random.uniform(0, 2 * math.pi)
                length = self.np_random.uniform(10, 20)
                end_pos = [start_pos[0] + math.cos(angle) * length, start_pos[1] + math.sin(angle) * length]

                end_pos[0] = np.clip(end_pos[0], 10, self.WIDTH - 10)
                end_pos[1] = np.clip(end_pos[1], 10, self.HEIGHT - 10)

                dist_to_player = math.hypot(end_pos[0] - self.player_nodes[0][0], end_pos[1] - self.player_nodes[0][1])
                if dist_to_player < comp['min_dist_to_player']:
                    reward -= 0.1 # Penalty for growing closer
                    comp['min_dist_to_player'] = dist_to_player

                comp['nodes'].append(end_pos)
                comp['segments'].append((start_node_idx, len(comp['nodes']) - 1))
        return reward

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1

            if p['lifespan'] <= 0:
                self.projectiles.remove(p)
                continue

            for i, comp in enumerate(self.competitors):
                if not comp['alive']: continue
                dist = math.hypot(p['pos'][0] - comp['core'][0], p['pos'][1] - comp['core'][1])
                if dist < comp['core_radius']:
                    damage = 25
                    comp['health'] -= damage
                    reward += 5 # Reward for hit
                    self._create_explosion(p['pos'], self.COLOR_SPORE, 20)
                    if p in self.projectiles: self.projectiles.remove(p)

                    if comp['health'] <= 0:
                        comp['alive'] = False
                        reward += 10 # Reward for elimination
                        self._create_explosion(comp['core'], comp['color'], 50)
                    break
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05 # slight gravity on particles
            p['life'] -= 1
            p['radius'] -= 0.1
            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _check_player_collision(self):
        player_core_pos = self.player_nodes[0]
        for comp in self.competitors:
            if not comp['alive']: continue
            for node_pos in comp['nodes']:
                dist = math.hypot(player_core_pos[0] - node_pos[0], player_core_pos[1] - node_pos[1])
                if dist < self.player_core_radius:
                    return True
        return False

    # --- Creation Helpers ---

    def _create_competitor(self, x, y, color, glow_color):
        return {
            'core': [x, y], 'nodes': [[x, y]], 'segments': [],
            'health': 100, 'max_health': 100,
            'color': color, 'glow_color': glow_color,
            'growth_cooldown': self.np_random.integers(0, 16), 'core_radius': 12,
            'alive': True, 'min_dist_to_player': math.hypot(x - self.WIDTH/2, y - (self.HEIGHT-25))
        }

    def _launch_spore(self):
        start_pos = list(self.player_nodes[0])
        target_pos = self.competitors[self.targeted_competitor_idx]['core']
        angle = math.atan2(target_pos[1] - start_pos[1], target_pos[0] - start_pos[0])
        speed = 5
        self.projectiles.append({
            'pos': start_pos,
            'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
            'radius': 5, 'lifespan': 150
        })

    def _create_particle(self, pos, color):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        return {
            'pos': list(pos),
            'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
            'life': self.np_random.integers(20, 41),
            'radius': self.np_random.uniform(2, 5),
            'color': color
        }

    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            self.particles.append(self._create_particle(pos, color))

    # --- Rendering ---

    def _render_game(self):
        # Render competitor networks + health auras
        for comp in self.competitors:
            if comp['alive']:
                self._render_network(comp['nodes'], comp['segments'], comp['color'], comp['core'], comp['core_radius'], comp['glow_color'])
                self._render_health_aura(comp)

        # Render player network
        self._render_network(self.player_nodes, self.player_segments, self.COLOR_PLAYER, self.player_nodes[0], self.player_core_radius, self.COLOR_PLAYER_GLOW)
        
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

        # Render projectiles
        for p in self.projectiles:
            self._draw_glow_circle(self.screen, p['pos'], p['radius'], self.COLOR_SPORE, (255, 255, 0, 100))

    def _render_ui(self):
        # Spore count
        spore_text = self.font_small.render(f"Spores: {self.spore_count}", True, self.COLOR_TEXT)
        self.screen.blit(spore_text, (10, 10))

        # Target indicator
        if self.competitors[self.targeted_competitor_idx]['alive']:
            target_pos = self.competitors[self.targeted_competitor_idx]['core']
            self._draw_target_indicator(target_pos)
        
        # Gravity indicator
        grav_start = (self.WIDTH - 30, 30)
        grav_end = (grav_start[0], grav_start[1] + 15 * self.gravity[1])
        pygame.draw.line(self.screen, self.COLOR_TEXT, grav_start, grav_end, 2)
        pygame.draw.polygon(self.screen, self.COLOR_TEXT, [(grav_end[0], grav_end[1] + 5 * self.gravity[1]), (grav_end[0]-4, grav_end[1]), (grav_end[0]+4, grav_end[1])])

        # Game Over / Victory text
        if self.game_over:
            msg = "VICTORY" if all(not c['alive'] for c in self.competitors) else "NETWORK CORRUPTED"
            text_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _create_soil_texture(self):
        # Use a seeded generator for deterministic texture
        rng = np.random.default_rng(seed=123)
        texture = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        texture.fill(self.COLOR_BG)
        for _ in range(200):
            pos = (rng.integers(0, self.WIDTH), rng.integers(0, self.HEIGHT))
            radius = rng.integers(1, 5)
            alpha = rng.integers(10, 41)
            color = (0, 0, 0, alpha)
            pygame.gfxdraw.filled_circle(texture, pos[0], pos[1], radius, color)
        return texture

    def _render_network(self, nodes, segments, color, core_pos, core_radius, glow_color):
        for start_idx, end_idx in segments:
            p1 = tuple(map(int, nodes[start_idx]))
            p2 = tuple(map(int, nodes[end_idx]))
            pygame.draw.aaline(self.screen, color, p1, p2, 1)
        for node in nodes:
            pygame.gfxdraw.filled_circle(self.screen, int(node[0]), int(node[1]), 2, color)
        self._draw_glow_circle(self.screen, core_pos, core_radius, color, glow_color)

    def _render_health_aura(self, comp):
        health_ratio = max(0, comp['health'] / comp['max_health'])
        radius = int(comp['core_radius'] * (1.5 + health_ratio * 1.5))
        alpha = int(30 + 70 * health_ratio)
        color = (*comp['color'], alpha)
        pygame.gfxdraw.filled_circle(self.screen, int(comp['core'][0]), int(comp['core'][1]), radius, color)

    def _draw_glow_circle(self, surface, pos, radius, color, glow_color):
        pos_int = (int(pos[0]), int(pos[1]))
        # Glow effect
        for i in range(3):
            r = int(radius * (1.2 + i * 0.4))
            alpha_glow = (glow_color[3] // (i + 1))
            pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], r, (*glow_color[:3], alpha_glow))
        # Main circle
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], int(radius), color)
        # Highlight
        highlight_pos = (pos_int[0] - int(radius*0.3), pos_int[1] - int(radius*0.3))
        pygame.gfxdraw.filled_circle(surface, highlight_pos[0], highlight_pos[1], int(radius*0.3), (255,255,255,100))

    def _draw_target_indicator(self, pos):
        size = 25
        bracket_size = 8
        color = self.COLOR_TARGET
        x, y = int(pos[0]), int(pos[1])

        # Top-left
        pygame.draw.lines(self.screen, color, False, [(x - size, y - size + bracket_size), (x - size, y - size), (x - size + bracket_size, y - size)], 2)
        # Top-right
        pygame.draw.lines(self.screen, color, False, [(x + size - bracket_size, y - size), (x + size, y - size), (x + size, y - size + bracket_size)], 2)
        # Bottom-left
        pygame.draw.lines(self.screen, color, False, [(x - size, y + size - bracket_size), (x - size, y + size), (x - size + bracket_size, y + size)], 2)
        # Bottom-right
        pygame.draw.lines(self.screen, color, False, [(x + size - bracket_size, y + size), (x + size, y + size), (x + size, y + size - bracket_size)], 2)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset(seed=42)
    done = False
    total_reward = 0
    
    # Mapping keyboard keys to MultiDiscrete actions
    # W/S: Gravity, A/D: Target, Space: Shoot
    key_map = {
        pygame.K_w: 1, pygame.K_s: 2,
        pygame.K_a: 3, pygame.K_d: 4,
    }

    # Create a window for rendering
    pygame.display.set_caption("Mycorrhizal Madness")
    window = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    print(GameEnv.user_guide)
    print("Use R to reset, Q to quit.")

    running = True
    while running:
        movement = 0
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset(seed=42)
                    total_reward = 0
                    print("--- Environment Reset ---")

        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement = move_val
                break
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = np.array([movement, space, shift])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the window
        # The observation is (H, W, C), but pygame surface wants (W, H)
        # and surfarray.make_surface expects (W, H, C)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        window.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset(seed=42)
            total_reward = 0
            
        clock.tick(env.FPS)

    env.close()