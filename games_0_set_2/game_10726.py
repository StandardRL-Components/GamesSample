import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:56:39.642596
# Source Brief: brief_00726.md
# Brief Index: 726
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An arcade-style Gymnasium environment where the player controls an organelle,
    navigating a cellular environment to reach the nucleus. The player must collect
    other organelles (reagents) to craft abilities, avoid hazardous proteins, and
    manage their health. The game features a dynamic camera, procedural level
    generation, and visually rich, bioluminescent graphics.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a cellular environment as an organelle, collecting reagents to craft abilities "
        "and avoiding hazards on your quest to reach the nucleus."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press Shift to cycle through crafts and "
        "Space to activate the selected craft."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WORLD_WIDTH = 2500
    WORLD_HEIGHT = 2500
    MAX_STEPS = 2000
    FPS = 30 # Assumed FPS for smooth motion

    # Colors (Bioluminescent Theme)
    COLOR_BG = (10, 20, 40)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_GLOW = (0, 255, 150, 50)
    COLOR_NUCLEUS = (255, 220, 50)
    COLOR_NUCLEUS_GLOW = (255, 220, 50, 70)
    COLOR_OBSTACLE = (255, 50, 100)
    COLOR_OBSTACLE_GLOW = (255, 50, 100, 60)
    COLOR_REAGENT = (50, 150, 255)
    COLOR_REAGENT_GLOW = (50, 150, 255, 60)
    COLOR_CYTOSKELETON = (255, 255, 255, 20)
    COLOR_GUIDE_LINE = (255, 255, 255, 30)
    COLOR_SHIELD = (100, 200, 255, 100)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_ACCENT = (0, 255, 150)
    COLOR_UI_INACTIVE = (100, 100, 120)

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
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # Game state variables are initialized in reset()
        self.player = None
        self.nucleus_pos = None
        self.obstacles = []
        self.reagents = []
        self.particles = []
        self.cytoskeleton = []
        self.camera_pos = np.array([0.0, 0.0])
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_dist_to_nucleus = 0
        self.difficulty_level = 0

        # Action state trackers
        self.prev_space_held = False
        self.prev_shift_held = False

        # Crafting system
        self.crafting_recipes = [
            {'name': 'BOOST', 'cost': {'reagent': 1}, 'effect': self._activate_boost, 'duration': 15, 'cooldown': 60},
            {'name': 'SHIELD', 'cost': {'reagent': 2}, 'effect': self._activate_shield, 'duration': 90, 'cooldown': 120},
        ]
        self.selected_craft_index = 0
        self.craft_cooldowns = [0] * len(self.crafting_recipes)
        self.active_effects = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.difficulty_level = 0

        # Player setup
        player_start_pos = np.array([self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2], dtype=float)
        self.player = {
            'pos': player_start_pos,
            'vel': np.array([0.0, 0.0]),
            'radius': 15,
            'max_health': 100,
            'health': 100,
            'reagents': 0
        }

        # Nucleus setup
        angle = self.np_random.uniform(0, 2 * math.pi)
        dist = self.WORLD_WIDTH / 2 - 200
        self.nucleus_pos = player_start_pos + np.array([math.cos(angle) * dist, math.sin(angle) * dist])

        # Procedural generation
        self.obstacles = []
        self.reagents = []
        self._spawn_entities(initial=True)
        
        # Cytoskeleton background
        self.cytoskeleton = []
        for _ in range(50):
            start = (self.np_random.uniform(0, self.WORLD_WIDTH), self.np_random.uniform(0, self.WORLD_HEIGHT))
            end = (start[0] + self.np_random.uniform(-200, 200), start[1] + self.np_random.uniform(-200, 200))
            self.cytoskeleton.append((start, end))

        self.particles = []
        self.last_dist_to_nucleus = self._get_dist_to_nucleus()

        # Reset action and crafting states
        self.prev_space_held = False
        self.prev_shift_held = False
        self.selected_craft_index = 0
        self.craft_cooldowns = [0] * len(self.crafting_recipes)
        self.active_effects = {}

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # --- Handle Input and Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] -= 1  # Up
        elif movement == 2: move_vec[1] += 1  # Down
        elif movement == 3: move_vec[0] -= 1  # Left
        elif movement == 4: move_vec[0] += 1  # Right
        
        player_speed = 6.0
        if 'boost' in self.active_effects:
            player_speed *= 2.0
        
        if np.linalg.norm(move_vec) > 0:
            self.player['vel'] += move_vec / np.linalg.norm(move_vec) * 0.8 * (player_speed / 6.0)
        
        # Crafting actions (on press, not hold)
        shift_pressed = shift_held and not self.prev_shift_held
        space_pressed = space_held and not self.prev_space_held

        if shift_pressed:
            self.selected_craft_index = (self.selected_craft_index + 1) % len(self.crafting_recipes)
            # sfx: UI_CYCLE_SOUND
        
        if space_pressed:
            craft = self.crafting_recipes[self.selected_craft_index]
            if self.player['reagents'] >= craft['cost']['reagent'] and self.craft_cooldowns[self.selected_craft_index] <= 0:
                self.player['reagents'] -= craft['cost']['reagent']
                craft['effect']()
                self.craft_cooldowns[self.selected_craft_index] = craft['cooldown']
                reward += 2.0
                # sfx: CRAFT_SUCCESS_SOUND
            else:
                # sfx: CRAFT_FAIL_SOUND
                pass

        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        # --- Update Game Logic ---
        # Update player
        self.player['pos'] += self.player['vel']
        self.player['vel'] *= 0.85 # Friction
        self._constrain_to_world(self.player)

        # Update obstacles
        for o in self.obstacles:
            o['pos'] += o['vel']
            if (o['pos'][0] < o['radius'] and o['vel'][0] < 0) or \
               (o['pos'][0] > self.WORLD_WIDTH - o['radius'] and o['vel'][0] > 0):
                o['vel'][0] *= -1
            if (o['pos'][1] < o['radius'] and o['vel'][1] < 0) or \
               (o['pos'][1] > self.WORLD_HEIGHT - o['radius'] and o['vel'][1] > 0):
                o['vel'][1] *= -1
        
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        
        # Update effects and cooldowns
        for effect in list(self.active_effects.keys()):
            self.active_effects[effect] -= 1
            if self.active_effects[effect] <= 0:
                del self.active_effects[effect]
        
        for i in range(len(self.craft_cooldowns)):
            if self.craft_cooldowns[i] > 0:
                self.craft_cooldowns[i] -= 1

        # --- Handle Collisions ---
        is_shielded = 'shield' in self.active_effects
        
        # Player vs Obstacles
        obstacles_to_remove = []
        for o in self.obstacles:
            if self._check_collision(self.player, o):
                if not is_shielded:
                    self.player['health'] -= o['damage']
                    reward -= 0.5
                    self._create_particles(self.player['pos'], 10, self.COLOR_OBSTACLE, 3, 5)
                    # sfx: PLAYER_DAMAGE_SOUND
                obstacles_to_remove.append(o) # Obstacle is destroyed on collision
        for o in obstacles_to_remove:
            self.obstacles.remove(o)

        # Player vs Reagents
        reagents_to_remove = []
        for r in self.reagents:
            if self._check_collision(self.player, r):
                self.player['reagents'] += 1
                reward += 1.0
                self.score += 10
                reagents_to_remove.append(r)
                self._create_particles(self.player['pos'], 15, self.COLOR_REAGENT, 2, 4)
                # sfx: COLLECT_REAGENT_SOUND
        for r in reagents_to_remove:
            self.reagents.remove(r)
        
        # --- Update Reward & Difficulty ---
        dist_to_nucleus = self._get_dist_to_nucleus()
        if dist_to_nucleus < self.last_dist_to_nucleus:
            reward += 0.1
        else:
            reward -= 0.05
        self.last_dist_to_nucleus = dist_to_nucleus

        if self.steps > 0 and self.steps % 200 == 0:
            self.difficulty_level += 1
            self._spawn_entities(initial=False)
            # sfx: DIFFICULTY_INCREASE_SOUND

        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if self.player['health'] <= 0:
            self.player['health'] = 0
            terminated = True
            self.game_over = True
            reward -= 100
            # sfx: PLAYER_DEATH_SOUND
        
        if dist_to_nucleus < self.player['radius'] + 30: # 30 is nucleus radius
            terminated = True
            self.game_over = True
            reward += 100
            self.score += 1000
            # sfx: VICTORY_SOUND
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        # Center camera on player
        self.camera_pos = self.player['pos'] - np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])

        self.screen.fill(self.COLOR_BG)
        
        # Render background elements
        self._render_cytoskeleton()
        self._render_guide_line()

        # Render game entities
        self._render_nucleus()
        for r in self.reagents: self._render_entity(r, self.COLOR_REAGENT, self.COLOR_REAGENT_GLOW)
        for o in self.obstacles: self._render_entity(o, self.COLOR_OBSTACLE, self.COLOR_OBSTACLE_GLOW)
        self._render_particles()
        self._render_player()

        # Render UI overlay
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player['health'],
            "reagents": self.player['reagents'],
            "distance_to_nucleus": self.last_dist_to_nucleus
        }

    # --- Helper & Rendering Methods ---

    def _to_screen_pos(self, world_pos):
        return world_pos - self.camera_pos

    def _render_entity(self, entity, color, glow_color):
        screen_pos = self._to_screen_pos(entity['pos'])
        
        # Check if on screen before drawing
        if -50 < screen_pos[0] < self.SCREEN_WIDTH + 50 and -50 < screen_pos[1] < self.SCREEN_HEIGHT + 50:
            # Glow effect
            pygame.gfxdraw.filled_circle(
                self.screen, int(screen_pos[0]), int(screen_pos[1]),
                int(entity['radius'] * 2.5), glow_color
            )
            # Main body
            pygame.gfxdraw.aacircle(self.screen, int(screen_pos[0]), int(screen_pos[1]), entity['radius'], color)
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), entity['radius'], color)

    def _render_player(self):
        screen_pos = self._to_screen_pos(self.player['pos'])
        
        # Health glow/outline
        health_ratio = self.player['health'] / self.player['max_health']
        glow_radius = int(self.player['radius'] * (1.5 + 1.5 * health_ratio))
        glow_alpha = int(30 + 70 * health_ratio)
        pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), glow_radius, (self.COLOR_PLAYER_GLOW[0], self.COLOR_PLAYER_GLOW[1], self.COLOR_PLAYER_GLOW[2], glow_alpha))

        # Shield effect
        if 'shield' in self.active_effects:
            shield_radius = self.player['radius'] + 5 + 3 * math.sin(self.steps * 0.2)
            pygame.gfxdraw.aacircle(self.screen, int(screen_pos[0]), int(screen_pos[1]), int(shield_radius), self.COLOR_SHIELD)
        
        # Boost effect
        if 'boost' in self.active_effects:
             self._create_particles(self.player['pos']-self.player['vel'], 1, self.COLOR_UI_ACCENT, 1, 2, 0.1)

        # Player body
        self._render_entity(self.player, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

    def _render_nucleus(self):
        screen_pos = self._to_screen_pos(self.nucleus_pos)
        if -100 < screen_pos[0] < self.SCREEN_WIDTH + 100 and -100 < screen_pos[1] < self.SCREEN_HEIGHT + 100:
            radius = 30
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), int(radius * 3.0), self.COLOR_NUCLEUS_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), int(radius * 2.0), self.COLOR_NUCLEUS_GLOW)
            # Main body
            pygame.gfxdraw.aacircle(self.screen, int(screen_pos[0]), int(screen_pos[1]), radius, self.COLOR_NUCLEUS)
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), radius, self.COLOR_NUCLEUS)
    
    def _render_cytoskeleton(self):
        for start, end in self.cytoskeleton:
            s_pos = self._to_screen_pos(start)
            e_pos = self._to_screen_pos(end)
            pygame.draw.aaline(self.screen, self.COLOR_CYTOSKELETON, s_pos, e_pos)

    def _render_guide_line(self):
        player_screen_pos = self._to_screen_pos(self.player['pos'])
        nucleus_screen_pos = self._to_screen_pos(self.nucleus_pos)
        pygame.draw.aaline(self.screen, self.COLOR_GUIDE_LINE, player_screen_pos, nucleus_screen_pos, 1)

    def _render_particles(self):
        for p in self.particles:
            screen_pos = self._to_screen_pos(p['pos'])
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), int(p['radius']), color)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Health Bar
        health_rect = pygame.Rect(10, 40, 200, 20)
        health_ratio = self.player['health'] / self.player['max_health']
        fill_width = int(health_rect.width * health_ratio)
        fill_rect = pygame.Rect(health_rect.left, health_rect.top, fill_width, health_rect.height)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, fill_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, health_rect, 2)

        # Reagents
        reagent_text = self.font_small.render(f"REAGENTS: {self.player['reagents']}", True, self.COLOR_REAGENT)
        self.screen.blit(reagent_text, (10, 70))

        # Crafting UI
        craft_y = self.SCREEN_HEIGHT - 40
        for i, craft in enumerate(self.crafting_recipes):
            is_selected = i == self.selected_craft_index
            can_afford = self.player['reagents'] >= craft['cost']['reagent']
            is_on_cooldown = self.craft_cooldowns[i] > 0
            
            color = self.COLOR_UI_TEXT
            if is_selected: color = self.COLOR_UI_ACCENT
            if not can_afford or is_on_cooldown: color = self.COLOR_UI_INACTIVE

            text = f"[{craft['name']}] cost: {craft['cost']['reagent']}"
            craft_text = self.font_small.render(text, True, color)
            
            text_pos = (10 + i * 150, craft_y)
            self.screen.blit(craft_text, text_pos)
            
            if is_on_cooldown:
                cooldown_ratio = self.craft_cooldowns[i] / craft['cooldown']
                bar_width = craft_text.get_width()
                bar_rect = pygame.Rect(text_pos[0], text_pos[1] + 18, bar_width * cooldown_ratio, 2)
                pygame.draw.rect(self.screen, self.COLOR_UI_INACTIVE, bar_rect)

    def _spawn_entities(self, initial=False):
        num_obstacles = 10 + self.difficulty_level * 5
        num_reagents = 5 + self.difficulty_level * 2
        
        if initial:
            num_obstacles = 20
            num_reagents = 10
        
        obstacle_speed = 1.0 + self.difficulty_level * 0.2

        for _ in range(num_obstacles):
            pos = self._get_random_spawn_pos(200)
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = np.array([math.cos(angle), math.sin(angle)]) * obstacle_speed
            self.obstacles.append({'pos': pos, 'vel': vel, 'radius': 10, 'damage': 10})

        for _ in range(num_reagents):
            pos = self._get_random_spawn_pos(100)
            self.reagents.append({'pos': pos, 'radius': 8})

    def _get_random_spawn_pos(self, min_dist_from_player):
        while True:
            pos = np.array([
                self.np_random.uniform(50, self.WORLD_WIDTH - 50),
                self.np_random.uniform(50, self.WORLD_HEIGHT - 50)
            ])
            if np.linalg.norm(pos - self.player['pos']) > min_dist_from_player:
                return pos

    def _constrain_to_world(self, entity):
        entity['pos'][0] = np.clip(entity['pos'][0], entity['radius'], self.WORLD_WIDTH - entity['radius'])
        entity['pos'][1] = np.clip(entity['pos'][1], entity['radius'], self.WORLD_HEIGHT - entity['radius'])

    def _check_collision(self, entity1, entity2):
        dist_sq = np.sum((entity1['pos'] - entity2['pos'])**2)
        radius_sum_sq = (entity1['radius'] + entity2['radius'])**2
        return dist_sq < radius_sum_sq

    def _get_dist_to_nucleus(self):
        return np.linalg.norm(self.player['pos'] - self.nucleus_pos)

    def _create_particles(self, pos, count, color, min_speed, max_speed, life_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            life = int(self.np_random.uniform(10, 20) * life_mult)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'life': life,
                'max_life': life, 'color': color, 'radius': self.np_random.integers(1, 4)
            })

    # --- Crafting Effects ---
    def _activate_boost(self):
        self.active_effects['boost'] = self.crafting_recipes[0]['duration']
        self._create_particles(self.player['pos'], 30, self.COLOR_UI_ACCENT, 2, 7)

    def _activate_shield(self):
        self.active_effects['shield'] = self.crafting_recipes[1]['duration']
        # sfx: SHIELD_ACTIVATE_SOUND
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Organelle Racer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Player Input ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'R' key
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)

    env.close()