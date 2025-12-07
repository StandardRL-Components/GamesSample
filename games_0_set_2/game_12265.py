import gymnasium as gym
import os
import pygame
import math
from collections import deque
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame.gfxdraw
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Deploy bots to terraform a chain of islands. Manage your energy to build, upgrade, "
        "and connect your network to achieve total terraformation."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to select an island. "
        "Press space to deploy a bot and shift to upgrade it."
    )
    auto_advance = True

    # Class-level variable for difficulty progression
    successful_episodes = 0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000

        # Colors
        self.COLOR_BG = (15, 25, 40)
        self.COLOR_WATER = (40, 80, 150)
        self.COLOR_LAND = (100, 200, 120)
        self.COLOR_SELECTOR = (0, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)
        self.COLOR_ENERGY = (255, 220, 50)
        self.COLOR_CONNECTION = (255, 255, 255)
        self.COLOR_BOT_BODY = (200, 200, 220)
        self.COLOR_BOT_LIGHT_ACTIVE = (50, 255, 50)
        self.COLOR_BOT_LIGHT_INACTIVE = (100, 100, 100)

        # Game Mechanics
        self.MIN_ISLANDS = 3
        self.MAX_ISLANDS = 8
        self.BOT_MAX_LEVEL = 5
        self.TERRAFORM_GOAL = 0.9
        self.CONNECTION_RANGE_FACTOR = 3.5  # Multiplier of avg radius

        # Costs
        self.COST_DEPLOY = {'energy': 50}
        self.COST_UPGRADE = {'energy': 75}
        self.COST_TERRAFORM_TICK = {'energy': 1}

        # Rewards
        self.REWARD_TERRAFORM_UNIT = 10.0  # Per 1% of total terraforming
        self.REWARD_CONNECT = 5.0
        self.REWARD_DEPLOY_BOT = 1.0
        self.REWARD_UPGRADE_BOT = 2.0
        self.REWARD_VICTORY = 100.0
        self.REWARD_FAILURE = -100.0

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
        self.font_s = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_m = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 32, bold=True)

        # Initialize state variables
        self.islands = []
        self.sorted_islands = []
        self.global_resources = {}
        self.selected_island_idx = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.last_total_terraform = 0.0
        self.last_connections = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        self._setup_level()

        self.last_total_terraform = sum(i['terraform_progress'] for i in self.islands)
        self.last_connections = len(self._get_active_connections())

        return self._get_observation(), self._get_info()

    def _setup_level(self):
        # Difficulty scales with successful episodes
        difficulty_tier = GameEnv.successful_episodes // 5
        num_islands = min(self.MIN_ISLANDS + difficulty_tier, self.MAX_ISLANDS)

        start_energy = max(200, 500 - difficulty_tier * 50)
        self.global_resources = {'energy': start_energy}

        self.islands = []

        # Generate islands in a grid-like pattern to avoid overlap
        grid_cols = math.ceil(math.sqrt(num_islands))
        grid_rows = math.ceil(num_islands / grid_cols)
        cell_w = self.WIDTH / grid_cols
        cell_h = self.HEIGHT / grid_rows

        positions = []
        for i in range(num_islands):
            row, col = divmod(i, grid_cols)
            x = col * cell_w + cell_w / 2 + self.np_random.uniform(-cell_w / 4, cell_w / 4)
            y = row * cell_h + cell_h / 2 + self.np_random.uniform(-cell_h / 4, cell_h / 4)
            positions.append((x, y))

        for i in range(num_islands):
            radius = self.np_random.uniform(25, 40)
            self.islands.append({
                'id': i,
                'pos': positions[i],
                'radius': radius,
                'terraform_progress': 0.0,
                'bot': None,  # e.g., {'level': 1}
            })

        # Sort islands for consistent selection control
        self.sorted_islands = sorted(self.islands, key=lambda i: (i['pos'][1], i['pos'][0]))
        self.selected_island_idx = 0

    def step(self, action):
        if self.game_over:
            # The environment has already terminated. Return a dummy step consistent with Gymnasium API.
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # 1. Unpack and handle player action
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        action_reward = self._handle_player_action(movement, space_pressed, shift_pressed)
        reward += action_reward

        # 2. Update game logic (bots work)
        self._update_game_logic()

        # 3. Calculate continuous rewards based on state change
        current_total_terraform = sum(i['terraform_progress'] for i in self.islands)
        terraform_delta = current_total_terraform - self.last_total_terraform
        reward += terraform_delta * self.REWARD_TERRAFORM_UNIT
        self.last_total_terraform = current_total_terraform

        current_connections = len(self._get_active_connections())
        connection_delta = current_connections - self.last_connections
        if connection_delta > 0:
            reward += connection_delta * self.REWARD_CONNECT
        self.last_connections = current_connections

        # 4. Check for termination conditions
        terminated = self._check_termination()
        if terminated:
            if self.victory:
                reward += self.REWARD_VICTORY
                GameEnv.successful_episodes += 1
            else:
                reward += self.REWARD_FAILURE

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_action(self, movement, deploy_pressed, upgrade_pressed):
        reward = 0.0
        if not self.sorted_islands:
            return 0.0

        num_islands = len(self.sorted_islands)
        grid_cols = math.ceil(math.sqrt(num_islands))

        # --- Movement ---
        if movement == 1:  # Up
            self.selected_island_idx = (self.selected_island_idx - grid_cols + num_islands) % num_islands
        elif movement == 2:  # Down
            self.selected_island_idx = (self.selected_island_idx + grid_cols) % num_islands
        elif movement == 3:  # Left
            self.selected_island_idx = (self.selected_island_idx - 1 + num_islands) % num_islands
        elif movement == 4:  # Right
            self.selected_island_idx = (self.selected_island_idx + 1) % num_islands

        selected_island = self.sorted_islands[self.selected_island_idx]

        # --- Deploy Bot (Space) ---
        if deploy_pressed:
            if selected_island['bot'] is None and self.global_resources['energy'] >= self.COST_DEPLOY['energy']:
                self.global_resources['energy'] -= self.COST_DEPLOY['energy']
                selected_island['bot'] = {'level': 1}
                reward += self.REWARD_DEPLOY_BOT

        # --- Upgrade Bot (Shift) ---
        if upgrade_pressed:
            if selected_island['bot'] is not None and selected_island['bot']['level'] < self.BOT_MAX_LEVEL:
                cost = self.COST_UPGRADE['energy'] * selected_island['bot']['level']
                if self.global_resources['energy'] >= cost:
                    self.global_resources['energy'] -= cost
                    selected_island['bot']['level'] += 1
                    reward += self.REWARD_UPGRADE_BOT

        return reward

    def _update_game_logic(self):
        # Bots consume energy to terraform
        for island in self.islands:
            if island['bot']:
                bot = island['bot']
                power = bot['level'] * 0.005  # Terraforms 0.5% per level per step
                cost = self.COST_TERRAFORM_TICK['energy'] * bot['level']

                if self.global_resources['energy'] >= cost:
                    self.global_resources['energy'] -= cost
                    island['terraform_progress'] = min(1.0, island['terraform_progress'] + power)

    def _check_termination(self):
        # Victory condition
        if self.islands:
            all_terraformed = all(i['terraform_progress'] >= self.TERRAFORM_GOAL for i in self.islands)
            all_connected = self._are_all_islands_connected()

            if all_terraformed and all_connected:
                self.game_over = True
                self.victory = True
                return True

        # --- Failure Condition ---
        # Can any existing bot work?
        can_terraform = False
        for island in self.islands:
            if island['bot']:
                cost = self.COST_TERRAFORM_TICK['energy'] * island['bot']['level']
                if self.global_resources['energy'] >= cost:
                    can_terraform = True
                    break

        # Can the player afford to deploy a new bot?
        can_deploy = self.global_resources['energy'] >= self.COST_DEPLOY['energy'] and any(
            i['bot'] is None for i in self.islands)

        # Can the player afford to upgrade any bot?
        can_upgrade = False
        for island in self.islands:
            if island['bot'] and island['bot']['level'] < self.BOT_MAX_LEVEL:
                cost = self.COST_UPGRADE['energy'] * island['bot']['level']
                if self.global_resources['energy'] >= cost:
                    can_upgrade = True
                    break

        # If no automatic progress is possible AND no player action is affordable, it's game over.
        if not can_terraform and not can_deploy and not can_upgrade:
            self.game_over = True
            return True

        # Max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True

        return False

    def _get_active_connections(self):
        connections = []
        active_islands = [i for i in self.islands if i['bot'] is not None]
        if not self.islands:
            return []
        avg_radius = sum(i['radius'] for i in self.islands) / len(self.islands) if self.islands else 1
        connection_range_sq = (avg_radius * self.CONNECTION_RANGE_FACTOR) ** 2

        for i in range(len(active_islands)):
            for j in range(i + 1, len(active_islands)):
                p1 = active_islands[i]['pos']
                p2 = active_islands[j]['pos']
                dist_sq = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
                if dist_sq < connection_range_sq:
                    connections.append((active_islands[i]['id'], active_islands[j]['id']))
        return connections

    def _are_all_islands_connected(self):
        if not self.islands:
            return True

        active_bot_islands = {i['id'] for i in self.islands if i['bot'] is not None}
        if len(active_bot_islands) < len(self.islands):
            return False  # All islands must have a bot to be part of the network

        connections = self._get_active_connections()
        adj = {i['id']: [] for i in self.islands}
        for u, v in connections:
            adj[u].append(v)
            adj[v].append(u)

        q = deque([self.islands[0]['id']])
        visited = {self.islands[0]['id']}
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)

        return len(visited) == len(self.islands)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_connections()
        self._render_islands()
        self._render_selector()

    def _render_connections(self):
        connections = self._get_active_connections()
        id_to_pos = {i['id']: i['pos'] for i in self.islands}
        pulse_progress = (self.steps % self.FPS) / self.FPS

        for u_id, v_id in connections:
            p1 = id_to_pos[u_id]
            p2 = id_to_pos[v_id]
            pygame.gfxdraw.aaline(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), self.COLOR_CONNECTION)

            # Pulsing effect
            pulse_x = int(p1[0] + (p2[0] - p1[0]) * pulse_progress)
            pulse_y = int(p1[1] + (p2[1] - p1[1]) * pulse_progress)
            pygame.gfxdraw.filled_circle(self.screen, pulse_x, pulse_y, 3, self.COLOR_ENERGY)
            pygame.gfxdraw.aacircle(self.screen, pulse_x, pulse_y, 3, self.COLOR_ENERGY)

    def _render_islands(self):
        for island in self.islands:
            x, y = int(island['pos'][0]), int(island['pos'][1])

            # Water base
            pygame.gfxdraw.filled_circle(self.screen, x, y, int(island['radius']), self.COLOR_WATER)
            pygame.gfxdraw.aacircle(self.screen, x, y, int(island['radius']), self.COLOR_WATER)

            # Terraformed land
            land_radius = int(island['radius'] * math.sqrt(island['terraform_progress']))
            if land_radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, x, y, land_radius, self.COLOR_LAND)
                pygame.gfxdraw.aacircle(self.screen, x, y, land_radius, self.COLOR_LAND)

            # Bot
            if island['bot']:
                self._render_bot(island)

    def _render_bot(self, island):
        x, y = int(island['pos'][0]), int(island['pos'][1])
        level = island['bot']['level']
        size = 8

        # Rotating body
        angle = math.radians(self.steps * 2)
        points = []
        for i in range(4):
            a = angle + math.pi / 2 * i
            px = x + math.cos(a) * size
            py = y + math.sin(a) * size
            points.append((px, py))
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_BOT_BODY)

        # Active light
        light_color = self.COLOR_BOT_LIGHT_ACTIVE if self.global_resources[
                                                         'energy'] >= self.COST_TERRAFORM_TICK[
                                                         'energy'] * level else self.COLOR_BOT_LIGHT_INACTIVE
        pygame.gfxdraw.filled_circle(self.screen, x, y, 4, light_color)

        # Level text
        self._draw_text(str(level), self.font_m, x, y, center=True)

    def _render_selector(self):
        if not self.sorted_islands: return

        selected_island = self.sorted_islands[self.selected_island_idx]
        x, y = int(selected_island['pos'][0]), int(selected_island['pos'][1])
        radius = int(selected_island['radius'] + 10 + 3 * math.sin(self.steps * 0.2))

        alpha = int(128 + 127 * math.sin(self.steps * 0.2))
        color = (*self.COLOR_SELECTOR, alpha)

        # Draw a glowing selector ring
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius - 1, color)

    def _render_ui(self):
        # Global resources
        energy_text = f"ENERGY: {int(self.global_resources['energy'])}"
        self._draw_text(energy_text, self.font_m, 10, 10)

        # Score and Steps
        score_text = f"SCORE: {self.score:.1f}"
        steps_text = f"STEP: {self.steps}/{self.MAX_STEPS}"
        self._draw_text(score_text, self.font_m, self.WIDTH - 150, 10)
        self._draw_text(steps_text, self.font_m, self.WIDTH - 150, 30)

        # Game Over message
        if self.game_over:
            msg = "VICTORY!" if self.victory else "FAILURE"
            color = self.COLOR_LAND if self.victory else self.COLOR_ENERGY
            self._draw_text(msg, self.font_l, self.WIDTH / 2, self.HEIGHT / 2 - 20, center=True, color=color)

    def _draw_text(self, text, font, x, y, color=None, center=False):
        if color is None: color = self.COLOR_TEXT
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surf = font.render(text, True, color)

        if center:
            text_rect = text_surf.get_rect(center=(x, y))
            shadow_rect = shadow_surf.get_rect(center=(x + 1, y + 1))
        else:
            text_rect = text_surf.get_rect(topleft=(x, y))
            shadow_rect = shadow_surf.get_rect(topleft=(x + 1, y + 1))

        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.global_resources.get('energy', 0),
            "islands": len(self.islands),
            "connections": len(self._get_active_connections()),
            "total_terraform_progress": sum(i['terraform_progress'] for i in self.islands)
        }

    def close(self):
        pygame.quit()