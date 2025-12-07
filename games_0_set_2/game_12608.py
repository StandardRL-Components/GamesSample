import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:50:17.054632
# Source Brief: brief_02608.md
# Brief Index: 2608
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Help a squirrel gather acorns to build its nest before winter arrives. "
        "Shrink to avoid the predator, but be quick!"
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to shrink and hide from the predator. "
        "Press shift at the nest to build."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_EPISODE_STEPS = 5000

    # Colors
    COLOR_BG = (25, 20, 35)
    COLOR_TREE = (60, 45, 70)
    COLOR_BUSH = (45, 35, 55)
    COLOR_SQUIRREL = (230, 190, 140)
    COLOR_SQUIRREL_GLOW = (255, 220, 180)
    COLOR_PREDATOR = (255, 80, 80)
    COLOR_PREDATOR_GLOW = (255, 120, 120)
    COLOR_ACORN = (140, 255, 140)
    COLOR_NEST_BASE = (100, 80, 60)
    COLOR_NEST_PROGRESS = (180, 160, 140)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SHADOW = (20, 20, 20)
    COLOR_UI_BAR_BG = (50, 50, 50)
    COLOR_UI_BAR_FILL = (100, 200, 100)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("sans-serif", 24)
        self.font_timer = pygame.font.SysFont("monospace", 32, bold=True)

        # --- Game State Variables (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        
        self.squirrel_pos = None
        self.squirrel_vel = None
        self.squirrel_size = None
        self.squirrel_target_size = None
        self.squirrel_size_change_speed = None
        self.is_small = None

        self.acorn_count = None
        self.total_acorns_collected = None
        self.acorns = None
        
        self.nest_pos = None
        self.nest_progress = None
        self.crafting_recipes = None
        self.unlocked_recipe_count = None

        self.predator_pos = None
        self.predator_speed = None
        self.predator_path = None
        self.predator_target_idx = None

        self.particles = None
        self.background_elements = None
        
        self.prev_space_held = None
        self.prev_shift_held = None

        # self.reset() # Removed to align with Gymnasium API best practices
        # self.validate_implementation() # Validation is for dev, not for __init__


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Squirrel
        self.squirrel_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        self.squirrel_vel = np.array([0.0, 0.0], dtype=float)
        self.squirrel_size = 12.0
        self.squirrel_target_size = 12.0
        self.squirrel_size_change_speed = 1.0
        self.is_small = False

        # Resources & Crafting
        self.acorn_count = 0
        self.total_acorns_collected = 0
        self.nest_pos = np.array([50, self.SCREEN_HEIGHT - 50], dtype=float)
        self.nest_progress = 0
        self.crafting_recipes = [
            {"cost": 10, "progress": 10}, {"cost": 15, "progress": 15},
            {"cost": 20, "progress": 20}, {"cost": 25, "progress": 25},
            {"cost": 30, "progress": 30}
        ]
        self.unlocked_recipe_count = 1

        # Predator
        self.predator_path = [
            np.array([50, 50]), np.array([self.SCREEN_WIDTH - 50, 50]),
            np.array([self.SCREEN_WIDTH - 50, self.SCREEN_HEIGHT - 50]),
            np.array([50, self.SCREEN_HEIGHT - 50])
        ]
        self.predator_pos = self.predator_path[0].copy().astype(float)
        self.predator_target_idx = 1
        self.predator_speed = 1.0

        # Items and Effects
        self.acorns = []
        self._spawn_acorns(15)
        self.particles = []
        self._generate_background()
        
        # Input state
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Process Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        self._handle_input(movement, space_pressed, shift_pressed)
        
        # --- Update Game Logic ---
        self._update_squirrel()
        self._update_predator()
        self._update_particles()
        self._update_game_progression()

        # --- Handle Interactions & Rewards ---
        collected_acorn = self._check_acorn_collection()
        if collected_acorn:
            reward += 0.1
            self.score += 0.1
            # sfx: acorn collect sound

        crafted = self._handle_crafting(shift_pressed)
        if crafted:
            reward += 1.0
            self.score += 1.0
            # sfx: crafting success sound

        predator_proximity_penalty = self._check_predator_proximity()
        reward += predator_proximity_penalty

        # --- Check Termination ---
        terminated = self._check_termination()
        
        # Terminal rewards
        if terminated:
            if self.nest_progress >= 100:
                reward = 100.0
                self.score += 100.0
            elif self.steps >= self.MAX_EPISODE_STEPS:
                reward = -10.0
                self.score -= 10.0
            else: # Caught by predator
                reward = -10.0
                self.score -= 10.0

        # --- Update Input State for next step ---
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Movement
        move_force = 2.0
        if movement == 1: self.squirrel_vel[1] -= move_force # Up
        elif movement == 2: self.squirrel_vel[1] += move_force # Down
        elif movement == 3: self.squirrel_vel[0] -= move_force # Left
        elif movement == 4: self.squirrel_vel[0] += move_force # Right
        
        # Size change
        if space_pressed:
            self.is_small = not self.is_small
            self.squirrel_target_size = 6.0 if self.is_small else 12.0
            # sfx: shrink/grow whoosh
            # Spawn particles for visual feedback
            self._spawn_particles(self.squirrel_pos, 20, self.COLOR_SQUIRREL_GLOW, 0.5, 2.0)

    def _update_squirrel(self):
        # Apply velocity and friction
        self.squirrel_vel *= 0.8  # Friction
        self.squirrel_pos += self.squirrel_vel

        # Clamp position to screen bounds
        self.squirrel_pos[0] = np.clip(self.squirrel_pos[0], self.squirrel_size, self.SCREEN_WIDTH - self.squirrel_size)
        self.squirrel_pos[1] = np.clip(self.squirrel_pos[1], self.squirrel_size, self.SCREEN_HEIGHT - self.squirrel_size)

        # Interpolate size
        size_diff = self.squirrel_target_size - self.squirrel_size
        self.squirrel_size += size_diff * 0.1 * self.squirrel_size_change_speed
    
    def _check_acorn_collection(self):
        collected_any = False
        for i in range(len(self.acorns) - 1, -1, -1):
            acorn_pos = self.acorns[i]
            dist = np.linalg.norm(self.squirrel_pos - acorn_pos)
            if dist < self.squirrel_size + 4: # 4 is acorn radius
                self.acorns.pop(i)
                self.acorn_count = min(100, self.acorn_count + 1)
                self.total_acorns_collected += 1
                collected_any = True
                self._spawn_particles(acorn_pos, 10, self.COLOR_ACORN, 0.3, 1.5)
        
        if len(self.acorns) < 10 and self.steps % 60 == 0:
            self._spawn_acorns(1)
        
        return collected_any

    def _handle_crafting(self, shift_pressed):
        if not shift_pressed:
            return False
            
        dist_to_nest = np.linalg.norm(self.squirrel_pos - self.nest_pos)
        if dist_to_nest > 50:
            # sfx: crafting fail sound
            return False

        # Find the next component to craft
        next_recipe_idx = self.nest_progress // 10 # Simplified logic
        if next_recipe_idx < len(self.crafting_recipes) and next_recipe_idx < self.unlocked_recipe_count:
            recipe = self.crafting_recipes[next_recipe_idx]
            if self.acorn_count >= recipe["cost"]:
                self.acorn_count -= recipe["cost"]
                self.nest_progress += recipe["progress"]
                self.nest_progress = min(100, self.nest_progress)
                self._spawn_particles(self.nest_pos, 30, self.COLOR_NEST_PROGRESS, 0.8, 3.0)
                return True
        
        # sfx: crafting fail sound
        return False

    def _update_predator(self):
        target = self.predator_path[self.predator_target_idx]
        direction = target - self.predator_pos
        dist = np.linalg.norm(direction)

        if dist < self.predator_speed:
            self.predator_pos = target.copy()
            self.predator_target_idx = (self.predator_target_idx + 1) % len(self.predator_path)
        else:
            self.predator_pos += (direction / dist) * self.predator_speed

    def _check_predator_proximity(self):
        dist_to_predator = np.linalg.norm(self.squirrel_pos - self.predator_pos)
        
        # Collision check
        if dist_to_predator < self.squirrel_size + 8 and not self.is_small: # 8 is predator radius
            self.game_over = True
            # sfx: player caught sound
            self._spawn_particles(self.squirrel_pos, 50, self.COLOR_PREDATOR, 1.0, 4.0)

        # Proximity penalty
        if dist_to_predator < 100 and not self.is_small:
            return -0.01
        return 0

    def _update_game_progression(self):
        # Predator speed increase
        self.predator_speed = 1.0 + (self.steps // 1000) * 0.05
        # Size change speed increase
        self.squirrel_size_change_speed = 1.0 + (self.total_acorns_collected // 100) * 0.1 # Faster progression
        # Recipe unlock
        self.unlocked_recipe_count = 1 + (self.total_acorns_collected // 50)
        self.unlocked_recipe_count = min(self.unlocked_recipe_count, len(self.crafting_recipes))

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.pop(i)
        
        # Add falling leaves
        if self.steps % 5 == 0 and len(self.particles) < 100:
            self.particles.append({
                'pos': np.array([random.uniform(0, self.SCREEN_WIDTH), 0.0]),
                'vel': np.array([random.uniform(-0.2, 0.2), random.uniform(0.3, 0.8)]),
                'life': random.randint(200, 400),
                'color': random.choice([(180, 90, 40), (210, 120, 50)]),
                'size': random.uniform(3, 6)
            })

    def _check_termination(self):
        if self.game_over:
            return True
        if self.nest_progress >= 100:
            self.game_over = True
            return True
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "acorn_count": self.acorn_count,
            "nest_progress": self.nest_progress,
        }

    # --- Rendering ---
    def _render_game(self):
        for bg_item in self.background_elements:
            pygame.draw.rect(self.screen, bg_item['color'], bg_item['rect'])
        
        self._render_particles()
        self._render_acorns()
        self._render_nest()
        self._render_predator()
        if not (self.game_over and self.nest_progress < 100):
            self._render_squirrel()
    
    def _render_squirrel(self):
        pos = (int(self.squirrel_pos[0]), int(self.squirrel_pos[1]))
        size = int(self.squirrel_size)
        
        # Glow effect
        glow_size = int(size * 1.5)
        glow_surface = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*self.COLOR_SQUIRREL_GLOW, 50), (glow_size, glow_size), glow_size)
        self.screen.blit(glow_surface, (pos[0] - glow_size, pos[1] - glow_size))
        
        # Main body
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_SQUIRREL)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_SQUIRREL)

    def _render_predator(self):
        pos = (int(self.predator_pos[0]), int(self.predator_pos[1]))
        size = 8
        
        # Glow
        glow_size = int(size * 2.0)
        glow_surface = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*self.COLOR_PREDATOR_GLOW, 80), (glow_size, glow_size), glow_size)
        self.screen.blit(glow_surface, (pos[0] - glow_size, pos[1] - glow_size))
        
        # Body
        points = [
            (pos[0], pos[1] - size),
            (pos[0] - size, pos[1] + size),
            (pos[0] + size, pos[1] + size)
        ]
        pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_PREDATOR)
        pygame.gfxdraw.aatrigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_PREDATOR)

    def _render_acorns(self):
        for acorn_pos in self.acorns:
            pos = (int(acorn_pos[0]), int(acorn_pos[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, self.COLOR_ACORN)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, self.COLOR_ACORN)

    def _render_nest(self):
        pos = (int(self.nest_pos[0]), int(self.nest_pos[1]))
        base_radius = 25
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], base_radius, self.COLOR_NEST_BASE)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], base_radius, self.COLOR_NEST_BASE)
        
        # Render progress as twigs
        for i in range(self.nest_progress):
            angle = i * 3.6 * (math.pi / 180) * 5 # Spread them out
            r1 = base_radius * 0.8 + (i/100) * 10
            r2 = base_radius + 5 + (i/100) * 15
            
            start_pos = (pos[0] + r1 * math.cos(angle), pos[1] + r1 * math.sin(angle))
            end_pos = (pos[0] + r2 * math.cos(angle), pos[1] + r2 * math.sin(angle))
            
            pygame.draw.line(self.screen, self.COLOR_NEST_PROGRESS, start_pos, end_pos, 2)
    
    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 100))))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / 100))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (int(p['pos'][0] - size), int(p['pos'][1] - size)))

    def _render_ui(self):
        # Acorn Count
        self._draw_text(f"Acorns: {self.acorn_count}", (10, 10))
        
        # Nest Progress Bar
        bar_width = 150
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        fill_width = (self.nest_progress / 100) * bar_width
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FILL, (bar_x, bar_y, fill_width, bar_height))
        self._draw_text("Nest", (bar_x + 5, bar_y - 2))

        # Winter Countdown
        time_left = self.MAX_EPISODE_STEPS - self.steps
        time_text = f"Winter in: {time_left}"
        text_surf = self.font_timer.render(time_text, True, self.COLOR_UI_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 25))
        
        shadow_surf = self.font_timer.render(time_text, True, self.COLOR_UI_SHADOW)
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)
        
        # Game Over / Win Text
        if self.game_over:
            if self.nest_progress >= 100:
                msg = "Nest Complete! You Survived!"
                color = self.COLOR_ACORN
            else:
                msg = "Game Over"
                color = self.COLOR_PREDATOR
            self._draw_text(msg, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2), font_size=48, center=True, color=color)

    # --- Utility Methods ---
    def _draw_text(self, text, pos, font_size=24, color=COLOR_UI_TEXT, shadow_color=COLOR_UI_SHADOW, center=False):
        font = self.font_ui if font_size == 24 else pygame.font.SysFont("sans-serif", font_size, bold=True)
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        
        if center:
            rect = text_surf.get_rect(center=pos)
        else:
            rect = text_surf.get_rect(topleft=pos)
            
        self.screen.blit(shadow_surf, (rect.x + 2, rect.y + 2))
        self.screen.blit(text_surf, rect)

    def _spawn_acorns(self, count):
        for _ in range(count):
            pos = self.np_random.uniform(low=[20, 20], high=[self.SCREEN_WIDTH - 20, self.SCREEN_HEIGHT - 20])
            # Ensure not spawning too close to nest initially
            if np.linalg.norm(pos - self.nest_pos) > 60:
                self.acorns.append(pos)

    def _spawn_particles(self, pos, count, color, speed_mult, life_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': self.np_random.integers(20, 41) * life_mult,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })
    
    def _generate_background(self):
        self.background_elements = []
        for _ in range(5): # Trees
            self.background_elements.append({
                'rect': pygame.Rect(self.np_random.integers(0, self.SCREEN_WIDTH-30), self.np_random.integers(0, self.SCREEN_HEIGHT-80), self.np_random.integers(20, 41), self.np_random.integers(60, 101)),
                'color': self.COLOR_TREE
            })
        for _ in range(8): # Bushes
            self.background_elements.append({
                'rect': pygame.Rect(self.np_random.integers(0, self.SCREEN_WIDTH-40), self.np_random.integers(0, self.SCREEN_HEIGHT-40), self.np_random.integers(30, 51), self.np_random.integers(20, 41)),
                'color': self.COLOR_BUSH
            })
            
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # --- Example Usage & Manual Play ---
    # Un-comment the line below to run with a display
    # os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for rendering
    try:
        render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Acorn Quest")
        clock = pygame.time.Clock()
        display_enabled = True
    except pygame.error:
        print("Pygame display could not be initialized (running in headless mode).")
        display_enabled = False

    total_reward = 0
    
    # Game loop for manual control
    running = True
    while running:
        movement = 0 # None
        space = 0 # Released
        shift = 0 # Released

        if display_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            action = [movement, space, shift]
        else: # If no display, just sample actions
            action = env.action_space.sample()


        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        if display_enabled:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            total_reward = 0
            obs, info = env.reset()
            if display_enabled:
                pygame.time.wait(2000) # Pause before restarting
            else: # If headless, exit after one episode
                running = False

        if display_enabled:
            clock.tick(30) # Run at 30 FPS

    env.close()