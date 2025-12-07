import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:52:04.803708
# Source Brief: brief_01965.md
# Brief Index: 1965
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An RTS-style Gymnasium environment where the player controls two worker units
    to gather resources and construct buildings, competing against an AI opponent.
    The goal is to build 3 structures before the AI does.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control two worker units in a real-time strategy game. Gather resources to construct "
        "buildings and out-build your AI opponent to achieve victory."
    )
    user_guide = (
        "Controls: Use arrow keys to move your selected worker. Press space to switch between "
        "workers. Hold shift near a build site to construct a building."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2500
    BUILDING_COST = 50
    BUILDING_HP = 100
    WIN_CONDITION = 3
    WORKER_SPEED = 4.0
    GATHER_RATE = 1
    BUILD_RATE = 2

    # --- COLORS ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (35, 45, 55)
    COLOR_PLAYER = (0, 255, 100)
    COLOR_PLAYER_LIGHT = (150, 255, 200)
    COLOR_AI = (255, 50, 50)
    COLOR_AI_LIGHT = (255, 150, 150)
    COLOR_RESOURCE = (50, 150, 255)
    COLOR_RESOURCE_LIGHT = (150, 200, 255)
    COLOR_SITE = (100, 100, 100)
    COLOR_SITE_BG = (50, 50, 50)
    COLOR_TEXT = (230, 230, 230)
    COLOR_WHITE = (255, 255, 255)
    COLOR_YELLOW = (255, 220, 0)

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("sans-serif", 50, bold=True)

        self.render_mode = render_mode
        self.game_over_message = ""
        
        # This will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_resources = 0
        self.ai_resources = 0
        self.player_workers = []
        self.ai_workers = []
        self.resource_nodes = []
        self.build_sites = []
        self.particles = []
        self.selected_worker_idx = 0
        self.prev_space_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.player_resources = 20
        self.ai_resources = 20
        self.selected_worker_idx = 0
        self.prev_space_held = False
        self.particles.clear()
        
        # --- Initialize Game Entities ---
        self.player_workers = [self._create_worker(is_player=True) for _ in range(2)]
        self.ai_workers = [self._create_worker(is_player=False) for _ in range(2)]

        self.resource_nodes = [
            {'pos': pygame.Vector2(self.np_random.integers(50, self.SCREEN_WIDTH-50), self.np_random.integers(50, self.SCREEN_HEIGHT-50)), 
             'amount': 150 + self.np_random.integers(0, 51)} 
            for _ in range(5)
        ]
        
        self.build_sites = [
            {'pos': pygame.Vector2(self.SCREEN_WIDTH * (i+1)/7, self.SCREEN_HEIGHT * j/3), 
             'owner': None, 'progress': 0} 
            for i in range(6) for j in [1, 2]
        ]

        return self._get_observation(), self._get_info()

    def _create_worker(self, is_player):
        return {
            'pos': pygame.Vector2(
                self.np_random.integers(20, self.SCREEN_WIDTH - 20), 
                self.np_random.integers(20, self.SCREEN_HEIGHT - 20)
            ),
            'target_pos': None,
            'state': 'idle', # idle, moving_to_resource, gathering, moving_to_build, building
            'target_entity': None,
            'angle': -90.0,
        }

    def step(self, action):
        reward = 0
        terminated = self.game_over
        truncated = False

        if not terminated:
            reward += self._handle_player_input(action)
            self._update_ai()
            reward += self._update_world()
            
            self.steps += 1
            self.score += reward

            terminated, win_status = self._check_termination()
            if win_status == "timeout":
                truncated = True
                terminated = False
            
            if terminated or truncated:
                self.game_over = True
                if win_status == "win":
                    terminal_reward = 100
                    self.game_over_message = "VICTORY"
                elif win_status == "loss":
                    terminal_reward = -100
                    self.game_over_message = "DEFEAT"
                else: # Draw/Timeout
                    terminal_reward = 0
                    self.game_over_message = "TIME OUT" if truncated else "STALEMATE"
                reward += terminal_reward
                self.score += terminal_reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_player_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        worker = self.player_workers[self.selected_worker_idx]

        # --- Worker Switching ---
        if space_held and not self.prev_space_held:
            self.selected_worker_idx = 1 - self.selected_worker_idx
            # SFX: UI_Switch.wav
        self.prev_space_held = space_held

        # --- Movement ---
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            worker['pos'] += move_vec.normalize() * self.WORKER_SPEED
            worker['pos'].x = np.clip(worker['pos'].x, 0, self.SCREEN_WIDTH)
            worker['pos'].y = np.clip(worker['pos'].y, 0, self.SCREEN_HEIGHT)
            worker['angle'] = move_vec.angle_to(pygame.Vector2(1, 0))
            worker['state'] = 'idle' # Manual movement overrides AI-like states

        # --- Building ---
        if shift_held:
            for site in self.build_sites:
                if site['owner'] != 'ai' and worker['pos'].distance_to(site['pos']) < 20:
                    worker['state'] = 'building'
                    worker['target_entity'] = site
                    break
        elif worker['state'] == 'building':
             worker['state'] = 'idle'
        
        return 0

    def _update_ai(self):
        for worker in self.ai_workers:
            if worker['state'] == 'idle':
                # If enough resources, prioritize building
                if self.ai_resources >= self.BUILDING_COST:
                    target_site = self._find_closest_target(worker['pos'], self.build_sites, lambda s: s['owner'] != 'player')
                    if target_site:
                        worker['target_entity'] = target_site
                        worker['target_pos'] = target_site['pos']
                        worker['state'] = 'moving_to_build'
                # Otherwise, gather resources
                else:
                    target_node = self._find_closest_target(worker['pos'], self.resource_nodes, lambda n: n['amount'] > 0)
                    if target_node:
                        worker['target_entity'] = target_node
                        worker['target_pos'] = target_node['pos']
                        worker['state'] = 'moving_to_resource'
            
            # Move towards target if exists
            if worker['target_pos']:
                move_vec = worker['target_pos'] - worker['pos']
                if move_vec.length() < 5:
                    worker['target_pos'] = None
                    if worker['state'] == 'moving_to_resource': worker['state'] = 'gathering'
                    if worker['state'] == 'moving_to_build': worker['state'] = 'building'
                else:
                    worker['pos'] += move_vec.normalize() * self.WORKER_SPEED
                    worker['angle'] = move_vec.angle_to(pygame.Vector2(1, 0))

    def _find_closest_target(self, pos, entity_list, condition):
        closest_dist = float('inf')
        closest_entity = None
        for entity in entity_list:
            if condition(entity):
                dist = pos.distance_to(entity['pos'])
                if dist < closest_dist:
                    closest_dist = dist
                    closest_entity = entity
        return closest_entity

    def _update_world(self):
        reward = 0
        all_workers = self.player_workers + self.ai_workers

        # --- Update Workers and Interactions ---
        for i, worker in enumerate(all_workers):
            is_player = i < len(self.player_workers)
            
            # --- Gathering ---
            if worker['state'] == 'gathering':
                node = worker['target_entity']
                if node and node['amount'] > 0:
                    node['amount'] -= self.GATHER_RATE
                    # SFX: Resource_Collect.wav
                    if is_player:
                        self.player_resources += self.GATHER_RATE
                        reward += 0.1
                        self._create_particles(worker['pos'], self.COLOR_RESOURCE, 1)
                    else:
                        self.ai_resources += self.GATHER_RATE
                    if node['amount'] <= 0:
                        worker['state'] = 'idle'
                else:
                    worker['state'] = 'idle'
            
            # --- Building ---
            if worker['state'] == 'building':
                site = worker['target_entity']
                if site:
                    resources = self.player_resources if is_player else self.ai_resources
                    owner = 'player' if is_player else 'ai'
                    if site['owner'] is None and resources >= self.BUILDING_COST:
                        site['owner'] = owner
                        if is_player: self.player_resources -= self.BUILDING_COST
                        else: self.ai_resources -= self.BUILDING_COST
                    
                    if site['owner'] == owner and site['progress'] < self.BUILDING_HP:
                        site['progress'] += self.BUILD_RATE
                        self._create_particles(site['pos'] + (random.uniform(-10, 10), random.uniform(-10, 10)), self.COLOR_YELLOW, 1, 0.5)
                        if site['progress'] >= self.BUILDING_HP:
                            site['progress'] = self.BUILDING_HP
                            # SFX: Building_Complete.wav
                            if is_player:
                                reward += 5.0
                            else:
                                reward -= 1.0 # AI completing a building is bad for player
                            worker['state'] = 'idle'
                else:
                    worker['state'] = 'idle'

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # --- Passive Income from Buildings ---
        if self.steps % 5 == 0:
            player_buildings = sum(1 for s in self.build_sites if s['owner'] == 'player' and s['progress'] >= self.BUILDING_HP)
            ai_buildings = sum(1 for s in self.build_sites if s['owner'] == 'ai' and s['progress'] >= self.BUILDING_HP)
            self.player_resources += player_buildings
            self.ai_resources += ai_buildings

        return reward

    def _check_termination(self):
        player_buildings = sum(1 for s in self.build_sites if s['owner'] == 'player' and s['progress'] >= self.BUILDING_HP)
        ai_buildings = sum(1 for s in self.build_sites if s['owner'] == 'ai' and s['progress'] >= self.BUILDING_HP)
        
        if player_buildings >= self.WIN_CONDITION:
            return True, "win"
        if ai_buildings >= self.WIN_CONDITION:
            return True, "loss"
        if self.steps >= self.MAX_STEPS:
            return True, "timeout"
        
        total_resources = sum(n['amount'] for n in self.resource_nodes)
        can_player_build = self.player_resources >= self.BUILDING_COST or any(s['owner'] == 'player' and s['progress'] < self.BUILDING_HP for s in self.build_sites)
        can_ai_build = self.ai_resources >= self.BUILDING_COST or any(s['owner'] == 'ai' and s['progress'] < self.BUILDING_HP for s in self.build_sites)
        
        if total_resources <= 0 and not can_player_build and not can_ai_build:
             return True, "draw"

        return False, ""

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw build sites
        for site in self.build_sites:
            rect = pygame.Rect(0, 0, 30, 30)
            rect.center = site['pos']
            pygame.draw.rect(self.screen, self.COLOR_SITE_BG, rect)
            if site['owner']:
                color = self.COLOR_PLAYER if site['owner'] == 'player' else self.COLOR_AI
                pygame.draw.rect(self.screen, color, rect, 2)
                
                progress_w = int(rect.width * (site['progress'] / self.BUILDING_HP))
                progress_rect = pygame.Rect(rect.left, rect.bottom + 2, progress_w, 4)
                pygame.draw.rect(self.screen, color, progress_rect)
            else:
                pygame.draw.rect(self.screen, self.COLOR_SITE, rect, 1, border_radius=2)

        # Draw resource nodes
        for node in self.resource_nodes:
            if node['amount'] > 0:
                size = int(10 + 10 * (node['amount'] / 200))
                pygame.gfxdraw.filled_circle(self.screen, int(node['pos'].x), int(node['pos'].y), size, self.COLOR_RESOURCE)
                pygame.gfxdraw.aacircle(self.screen, int(node['pos'].x), int(node['pos'].y), size, self.COLOR_RESOURCE_LIGHT)

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            p_color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, p_color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, p['pos'] - (p['size'], p['size']), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw workers
        for i, worker in enumerate(self.player_workers):
            self._draw_triangle(self.screen, worker['pos'], worker['angle'], 12, self.COLOR_PLAYER)
            if i == self.selected_worker_idx and not self.game_over:
                self._draw_glow_circle(self.screen, worker['pos'], 18, self.COLOR_WHITE)
        for worker in self.ai_workers:
            self._draw_triangle(self.screen, worker['pos'], worker['angle'], 12, self.COLOR_AI)

    def _render_ui(self):
        # Player resources
        player_res_text = self.font_ui.render(f"RES: {int(self.player_resources)}", True, self.COLOR_PLAYER)
        self.screen.blit(player_res_text, (10, 10))
        
        # AI resources
        ai_res_text = self.font_ui.render(f"AI RES: {int(self.ai_resources)}", True, self.COLOR_AI)
        self.screen.blit(ai_res_text, (self.SCREEN_WIDTH - ai_res_text.get_width() - 10, 10))

        # Player buildings
        player_buildings = sum(1 for s in self.build_sites if s['owner'] == 'player' and s['progress'] >= self.BUILDING_HP)
        player_build_text = self.font_ui.render(f"BLD: {player_buildings}/{self.WIN_CONDITION}", True, self.COLOR_PLAYER)
        self.screen.blit(player_build_text, (10, 30))
        
        # AI buildings
        ai_buildings = sum(1 for s in self.build_sites if s['owner'] == 'ai' and s['progress'] >= self.BUILDING_HP)
        ai_build_text = self.font_ui.render(f"AI BLD: {ai_buildings}/{self.WIN_CONDITION}", True, self.COLOR_AI)
        self.screen.blit(ai_build_text, (self.SCREEN_WIDTH - ai_build_text.get_width() - 10, 30))
        
        # Game Over Message
        if self.game_over:
            color = self.COLOR_PLAYER if self.game_over_message == "VICTORY" else self.COLOR_AI
            msg_text = self.font_msg.render(self.game_over_message, True, color)
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_resources": self.player_resources,
            "ai_resources": self.ai_resources,
            "player_buildings": sum(1 for s in self.build_sites if s['owner'] == 'player' and s['progress'] >= self.BUILDING_HP),
            "ai_buildings": sum(1 for s in self.build_sites if s['owner'] == 'ai' and s['progress'] >= self.BUILDING_HP),
        }

    def _draw_triangle(self, surface, pos, angle_deg, size, color):
        angle_rad = math.radians(angle_deg)
        points = [
            (pos.x + size * math.cos(angle_rad), pos.y - size * math.sin(angle_rad)),
            (pos.x + size/2 * math.cos(angle_rad + 2.5), pos.y - size/2 * math.sin(angle_rad + 2.5)),
            (pos.x + size/2 * math.cos(angle_rad - 2.5), pos.y - size/2 * math.sin(angle_rad - 2.5)),
        ]
        pygame.gfxdraw.aapolygon(surface, [(int(p[0]), int(p[1])) for p in points], color)
        pygame.gfxdraw.filled_polygon(surface, [(int(p[0]), int(p[1])) for p in points], color)
    
    def _draw_glow_circle(self, surface, pos, radius, color):
        for i in range(radius, 0, -2):
            alpha = 100 - (i / radius) * 100
            pygame.gfxdraw.aacircle(surface, int(pos.x), int(pos.y), i, (*color, int(alpha)))
    
    def _create_particles(self, pos, color, count, speed_scale=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 1.5) * speed_scale
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'life': random.randint(20, 40),
                'max_life': 40,
                'color': color,
                'size': random.randint(1, 3)
            })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Example Usage ---
    # Un-comment the following line to run in a window
    # os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    env = GameEnv()
    
    # --- Manual Play ---
    # To play, you need a window. The testing environment runs headless.
    if "dummy" not in os.environ.get("SDL_VIDEODRIVER", ""):
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("RTS Worker Environment")
        
        obs, info = env.reset()
        done = False
        clock = pygame.time.Clock()
        
        while not done:
            # --- Action Mapping for Human ---
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

            # --- Gym Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- Rendering ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()

            clock.tick(30) # Limit frame rate for playability
    else:
        print("Running in headless mode. No window will be displayed.")
        # Basic test in headless mode
        obs, info = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print("Episode finished.")
                obs, info = env.reset()
        print("Headless test run complete.")

    env.close()