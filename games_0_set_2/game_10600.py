import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:38:03.252010
# Source Brief: brief_00600.md
# Brief Index: 600
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a cyberpunk resource scavenging game.
    The player controls a robot to collect parts using magnetism while avoiding patrols.
    The goal is to craft the ultimate 'Purifier' tool.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "In a cyberpunk world, control a robot to scavenge for parts using magnetism, "
        "craft tools, and avoid patrols to build the ultimate 'Purifier'."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Hold space to attract parts with your magnet, "
        "and hold shift to repel them."
    )
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 150)
    COLOR_PATROL = (255, 20, 20)
    COLOR_PATROL_GLOW = (150, 20, 20)
    COLOR_MAGNET_ATTRACT = (255, 255, 0, 100)
    COLOR_MAGNET_REPEL = (255, 0, 255, 100)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BG = (30, 30, 60, 150)
    COLOR_HEALTH_BAR = (0, 200, 0)
    COLOR_HEALTH_BAR_BG = (80, 0, 0)
    COLOR_COMBO_BAR = (255, 165, 0)
    
    PART_COLORS = {
        "scrap": (100, 100, 200),
        "circuitry": (50, 200, 50),
        "plastic": (200, 200, 50)
    }

    # Screen Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game Parameters
    PLAYER_SIZE = 10
    PLAYER_SPEED = 3.0
    PATROL_SIZE = 8
    PATROL_INITIAL_SPEED = 1.0
    PATROL_SPEED_INCREASE_INTERVAL = 500
    PATROL_SPEED_INCREASE_AMOUNT = 0.05
    PATROL_SPAWN_INTERVAL = 1000
    PART_SIZE = 4
    MAX_PARTS = 30
    MAGNET_FORCE = 0.5
    MAGNET_RANGE = 150
    MAX_STEPS = 5000
    COMBO_TIMEOUT = 90 # 3 seconds at 30 FPS

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- Game State Initialization ---
        self.player_pos = pygame.math.Vector2(0, 0)
        self.health = 100.0
        self.inventory = {}
        self.patrols = []
        self.parts = []
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.combo_count = 0
        self.combo_timer = 0
        self.crafted_tools = set()
        self.unlocked_recipes = set()
        self.current_magnet_strength = self.MAGNET_FORCE
        self.current_player_speed = self.PLAYER_SPEED
        self.last_reward = 0.0
        
        self._setup_crafting_system()
        # self.reset() is called by the wrapper, no need to call it here
        
    def _setup_crafting_system(self):
        self.crafting_recipes = {
            'Magnet_Boost_1': {'scrap': 5, 'circuitry': 2, 'unlock_req': 0},
            'Speed_Boost_1': {'scrap': 3, 'plastic': 5, 'unlock_req': 10},
            'Magnet_Boost_2': {'scrap': 10, 'circuitry': 8, 'plastic': 5, 'unlock_req': 25},
            'Purifier': {'scrap': 20, 'circuitry': 15, 'plastic': 10, 'unlock_req': 50}
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.health = 100.0
        
        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        self.inventory = { "scrap": 0, "circuitry": 0, "plastic": 0 }
        self.crafted_tools = set()
        self.unlocked_recipes = set()
        
        self.current_magnet_strength = self.MAGNET_FORCE
        self.current_player_speed = self.PLAYER_SPEED

        self.patrols = []
        self._spawn_patrol()
        
        self.parts = []
        while len(self.parts) < self.MAX_PARTS:
            self._spawn_part()
            
        self.particles = []
        self.combo_count = 0
        self.combo_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.last_action = action
        self.last_reward = 0.0
        self.steps += 1
        
        # --- Handle Actions ---
        self._handle_input(action)
        
        # --- Update Game State ---
        self._update_patrols()
        self._update_parts(action)
        self._update_combo()
        self._update_particles()
        
        # --- Handle Collisions ---
        self._handle_collisions()
        
        # --- Handle Crafting & Progression ---
        self._check_recipe_unlocks()
        self._check_crafting()
        self._update_difficulty()
        
        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over: # First frame of termination
            if self.health <= 0:
                self.last_reward -= 100.0
            elif "Purifier" in self.crafted_tools:
                self.last_reward += 100.0
            self.game_over = True
        
        self.score += self.last_reward
        
        return (
            self._get_observation(),
            self.last_reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1  # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1  # Right
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.current_player_speed
        
        # Clamp player position to screen bounds
        self.player_pos.x = max(self.PLAYER_SIZE, min(self.SCREEN_WIDTH - self.PLAYER_SIZE, self.player_pos.x))
        self.player_pos.y = max(self.PLAYER_SIZE, min(self.SCREEN_HEIGHT - self.PLAYER_SIZE, self.player_pos.y))

    def _update_patrols(self):
        for patrol in self.patrols:
            patrol['pos'] += patrol['dir'] * patrol['speed']
            if patrol['pos'].x <= self.PATROL_SIZE or patrol['pos'].x >= self.SCREEN_WIDTH - self.PATROL_SIZE:
                patrol['dir'].x *= -1
            if patrol['pos'].y <= self.PATROL_SIZE or patrol['pos'].y >= self.SCREEN_HEIGHT - self.PATROL_SIZE:
                patrol['dir'].y *= -1

    def _update_parts(self, action):
        _, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Magnetism
        if space_held or shift_held:
            force_direction = 1 if space_held else -1 # 1 for attract, -1 for repel
            for part in self.parts:
                dist_vec = self.player_pos - part['pos']
                dist = dist_vec.length()
                if 0 < dist < self.MAGNET_RANGE:
                    force = dist_vec.normalize() * self.current_magnet_strength * force_direction
                    part['vel'] += force
        
        # Update part positions
        for part in self.parts:
            part['pos'] += part['vel']
            part['vel'] *= 0.9 # Damping
            # Clamp to screen
            part['pos'].x = max(self.PART_SIZE, min(self.SCREEN_WIDTH - self.PART_SIZE, part['pos'].x))
            part['pos'].y = max(self.PART_SIZE, min(self.SCREEN_HEIGHT - self.PART_SIZE, part['pos'].y))

    def _update_combo(self):
        if self.combo_timer > 0:
            self.combo_timer -= 1
        else:
            self.combo_count = 0

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _handle_collisions(self):
        # Player-Patrol collision
        for patrol in self.patrols:
            if self.player_pos.distance_to(patrol['pos']) < self.PLAYER_SIZE + self.PATROL_SIZE:
                self.health = max(0, self.health - 0.5)
                self.last_reward -= 0.5
                self.combo_count = 0
                self.combo_timer = 0
                self._create_spark_effect(self.player_pos, self.COLOR_PATROL)
                # Sound: player_hit.wav

        # Player-Part collision
        for part in self.parts[:]:
            if self.player_pos.distance_to(part['pos']) < self.PLAYER_SIZE + self.PART_SIZE:
                self.inventory[part['type']] += 1
                self.parts.remove(part)
                self.last_reward += 0.1
                self.combo_timer = self.COMBO_TIMEOUT
                self.combo_count += 1
                self._create_spark_effect(part['pos'], self.PART_COLORS[part['type']])
                # Sound: collect_part.wav
        
        # Respawn parts if needed
        while len(self.parts) < self.MAX_PARTS:
            self._spawn_part()

    def _check_recipe_unlocks(self):
        total_parts = sum(self.inventory.values())
        for name, recipe in self.crafting_recipes.items():
            if name not in self.unlocked_recipes and total_parts >= recipe['unlock_req']:
                self.unlocked_recipes.add(name)
                self.last_reward += 5.0
                # Sound: recipe_unlocked.wav

    def _check_crafting(self):
        for name, recipe in self.crafting_recipes.items():
            if name in self.unlocked_recipes and name not in self.crafted_tools:
                can_craft = all(self.inventory[item] >= amount for item, amount in recipe.items() if item in self.inventory)
                if can_craft:
                    for item, amount in recipe.items():
                        if item in self.inventory:
                            self.inventory[item] -= amount
                    self.crafted_tools.add(name)
                    self._apply_craft_bonus(name)
                    self.last_reward += 1.0
                    self._create_spark_effect(self.player_pos, self.COLOR_UI_TEXT, 50)
                    # Sound: craft_tool.wav

    def _apply_craft_bonus(self, tool_name):
        if tool_name == 'Magnet_Boost_1':
            self.current_magnet_strength *= 1.5
        elif tool_name == 'Speed_Boost_1':
            self.current_player_speed *= 1.25
        elif tool_name == 'Magnet_Boost_2':
            self.current_magnet_strength *= 1.5
        # Purifier is the win condition, no direct bonus

    def _update_difficulty(self):
        # Increase patrol speed
        current_patrol_speed = self.PATROL_INITIAL_SPEED + self.PATROL_SPEED_INCREASE_AMOUNT * (self.steps // self.PATROL_SPEED_INCREASE_INTERVAL)
        for p in self.patrols:
            p['speed'] = current_patrol_speed

        # Spawn new patrols
        if self.steps > 0 and self.steps % self.PATROL_SPAWN_INTERVAL == 0:
            if len(self.patrols) < 5: # Max 5 patrols
                self._spawn_patrol()

    def _check_termination(self):
        return self.health <= 0 or "Purifier" in self.crafted_tools

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_effects()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.health,
            "inventory": self.inventory.copy(),
            "combo": self.combo_count,
            "crafted_tools": list(self.crafted_tools)
        }

    # --- Spawning Methods ---
    def _spawn_patrol(self):
        edge = self.np_random.integers(4)
        if edge == 0: pos = pygame.math.Vector2(0, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        elif edge == 1: pos = pygame.math.Vector2(self.SCREEN_WIDTH, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        elif edge == 2: pos = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), 0)
        else: pos = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT)
        
        direction = pygame.math.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize()
        
        self.patrols.append({
            'pos': pos,
            'dir': direction,
            'speed': self.PATROL_INITIAL_SPEED
        })
        
    def _spawn_part(self):
        part_type = self.np_random.choice(list(self.PART_COLORS.keys()))
        self.parts.append({
            'pos': pygame.math.Vector2(self.np_random.uniform(20, self.SCREEN_WIDTH - 20), self.np_random.uniform(20, self.SCREEN_HEIGHT - 20)),
            'type': part_type,
            'vel': pygame.math.Vector2(0, 0)
        })

    def _create_spark_effect(self, pos, color, count=20):
        for _ in range(count):
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': pygame.math.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)),
                'life': self.np_random.integers(10, 30),
                'color': color
            })

    # --- Rendering Methods ---
    def _render_background_effects(self):
        # Faint building silhouettes
        # Note: This uses a non-deterministic random call inside render.
        # For a fully deterministic environment, this should be seeded or based on game state.
        for i in range(5):
            x = int(self.np_random.uniform(0, self.SCREEN_WIDTH))
            h = int(self.np_random.uniform(50, 200))
            w = int(self.np_random.uniform(20, 80))
            pygame.draw.rect(self.screen, (20, 15, 40), (x, self.SCREEN_HEIGHT - h, w, h))

    def _render_game(self):
        action = [0,0,0] 
        if hasattr(self, 'last_action'):
             action = self.last_action
        
        space_held, shift_held = action[1] == 1, action[2] == 1

        # Magnetism field visual
        if space_held:
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.MAGNET_RANGE, self.COLOR_MAGNET_ATTRACT)
        elif shift_held:
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.MAGNET_RANGE, self.COLOR_MAGNET_REPEL)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color_with_alpha = (*p['color'], alpha)
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), 2, color_with_alpha)
            except TypeError: # Handle cases where color might not have alpha
                 pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), 2, (*p['color'][:3], alpha))


        # Parts
        for part in self.parts:
            pos_int = (int(part['pos'].x), int(part['pos'].y))
            color = self.PART_COLORS[part['type']]
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PART_SIZE, color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PART_SIZE, color)

        # Patrols
        for patrol in self.patrols:
            pos_int = (int(patrol['pos'].x), int(patrol['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PATROL_SIZE + 5, self.COLOR_PATROL_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PATROL_SIZE, self.COLOR_PATROL)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PATROL_SIZE, self.COLOR_PATROL)
            
        # Player
        player_pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_SIZE + 8, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_SIZE, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_SIZE, self.COLOR_PLAYER)

    def _render_ui(self):
        # --- Health Bar ---
        health_rect_bg = pygame.Rect(10, 10, 200, 20)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, health_rect_bg)
        health_width = max(0, int(200 * (self.health / 100.0)))
        health_rect = pygame.Rect(10, 10, health_width, 20)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, health_rect)
        self._draw_text(f"HP: {int(self.health)}/100", (110, 20), center=True)

        # --- Score ---
        self._draw_text(f"SCORE: {int(self.score)}", (self.SCREEN_WIDTH / 2, 20), center=True)

        # --- Combo Meter ---
        combo_rect_bg = pygame.Rect(self.SCREEN_WIDTH - 160, 10, 150, 20)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, combo_rect_bg)
        if self.combo_timer > 0:
            combo_width = int(150 * (self.combo_timer / self.COMBO_TIMEOUT))
            combo_rect = pygame.Rect(self.SCREEN_WIDTH - 160, 10, combo_width, 20)
            pygame.draw.rect(self.screen, self.COLOR_COMBO_BAR, combo_rect)
        if self.combo_count > 1:
            self._draw_text(f"{self.combo_count}x COMBO", (self.SCREEN_WIDTH - 85, 20), center=True)

        # --- Inventory & Crafting UI ---
        ui_panel = pygame.Surface((200, 120), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self._draw_text("INVENTORY", (100, 10), font=self.font_large, surface=ui_panel, center=True)
        y_offset = 35
        for item, count in self.inventory.items():
            color = self.PART_COLORS[item]
            pygame.draw.circle(ui_panel, color, (20, y_offset), 5)
            self._draw_text(f"{item.capitalize()}: {count}", (30, y_offset - 8), surface=ui_panel)
            y_offset += 20
        
        goal_text = "Goal: Craft Purifier"
        if "Purifier" in self.crafted_tools:
            goal_text = "PURIFIER CRAFTED!"
        elif "Purifier" in self.unlocked_recipes:
            goal_text = "Goal: Craft Purifier [READY]"

        self._draw_text(goal_text, (100, 95), surface=ui_panel, center=True)
        self.screen.blit(ui_panel, (self.SCREEN_WIDTH - 210, self.SCREEN_HEIGHT - 130))

    def _draw_text(self, text, pos, color=COLOR_UI_TEXT, font=None, surface=None, center=False):
        if font is None: font = self.font_small
        if surface is None: surface = self.screen
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        surface.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    try:
        os.environ.pop("SDL_VIDEODRIVER")
    except KeyError:
        pass
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Cyberpunk Scavenger")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    while running:
        # --- Human Input ---
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- Resetting Environment ---")
                obs, info = env.reset()
                total_reward = 0.0

        if terminated or truncated:
            print(f"--- Episode Finished ---")
            print(f"Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            print("Press 'R' to restart.")
            # Simple wait for 'R' to restart or quit
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0.0
                        wait_for_reset = False


        clock.tick(30) # Run at 30 FPS

    env.close()