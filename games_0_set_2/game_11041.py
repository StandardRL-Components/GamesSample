import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:24:05.610877
# Source Brief: brief_01041.md
# Brief Index: 1041
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class LifeForm:
    def __init__(self, pos, shape_sides, color, size):
        self.pos = pygame.Vector2(pos)
        self.shape_sides = shape_sides
        self.color = color
        self.size = size
        self.angle = random.uniform(0, 2 * math.pi)
        self.rotation_speed = random.uniform(-0.02, 0.02)
        self.pulse_timer = random.uniform(0, 2 * math.pi)
        self.pulse_speed = random.uniform(0.05, 0.1)
        self.nearest_pool_idx = -1

    def update(self):
        self.angle += self.rotation_speed
        self.pulse_timer += self.pulse_speed

    def get_current_size(self):
        pulse = (math.sin(self.pulse_timer) + 1) / 2  # 0 to 1
        return self.size * (0.95 + 0.1 * pulse)

class Particle:
    def __init__(self, pos, vel, start_color, end_color, life):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.start_color = start_color
        self.end_color = end_color
        self.life = life
        self.max_life = life

    def update(self):
        self.pos += self.vel
        self.vel *= 0.98  # friction
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            t = self.life / self.max_life
            color = (
                int(self.start_color[0] * t + self.end_color[0] * (1 - t)),
                int(self.start_color[1] * t + self.end_color[1] * (1 - t)),
                int(self.start_color[2] * t + self.end_color[2] * (1 - t)),
            )
            radius = int(t * 3)
            if radius > 0:
                pygame.draw.circle(surface, color, self.pos, radius)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Solve word puzzles to create new life forms and strategically place them to grow a digital ecosystem. "
        "Manage your resources carefully to prevent collapse."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press [SHIFT] to solve the anagram and generate a life form, "
        "then press [SPACE] to place it."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    TARGET_FPS = 30
    MAX_STEPS = 6000 # 200 seconds at 30fps

    COLOR_BG = (26, 28, 44)
    COLOR_GRID = (40, 42, 60)
    COLOR_RESOURCE = (0, 255, 150)
    COLOR_RESOURCE_GLOW = (0, 100, 60, 50)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 15)
    COLOR_PANEL = (35, 38, 58, 180)
    
    LIFE_FORM_COLORS = [
        (255, 0, 128), (0, 255, 255), (255, 128, 0), 
        (128, 0, 255), (0, 255, 0)
    ]
    LIFE_FORM_SHAPES = [3, 4, 5, 6, 8] # Sides

    VICTORY_SCORE = 20
    CURSOR_SPEED = 8

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 48)

        self.word_list = ["CIRCLE", "SQUARE", "LINE", "CELL", "GROW", "SEED", "GRID", "FORM", "LIFE", "POINT"]
        
        self.life_forms = []
        self.resource_pools = []
        self.particles = []
        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.pulse_timer = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = "" # "VICTORY" or "DEFEAT"
        
        self.life_forms = []
        self._generate_resource_pools()
        self.particles = []
        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        
        self.game_phase = "ANAGRAM" # "ANAGRAM" or "PLACEMENT"
        self.current_life_form_to_place = None
        self._generate_anagram()

        self.successful_placements = 0
        self.depletion_multiplier = 1.0
        
        self.prev_space_held = True # Force release on first step
        self.prev_shift_held = True

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.pulse_timer += 0.1
        reward = 0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        # --- Handle Actions & Rewards ---
        if self.game_phase == "ANAGRAM":
            if shift_pressed:
                # Event: Solved anagram
                reward += 5.0
                self._generate_life_form_for_placement()
                self.game_phase = "PLACEMENT"
                # SFX: success_chime.wav
            elif space_pressed:
                # Penalty: Tried to place with no form
                reward -= 1.0
                # SFX: error_buzz.wav
        
        elif self.game_phase == "PLACEMENT":
            self._move_cursor(movement)
            if space_pressed:
                # Event: Attempt to place
                placed, reason = self._place_life_form()
                if placed:
                    reward += 1.0
                    self.successful_placements += 1
                    if self.successful_placements > 0 and self.successful_placements % 5 == 0:
                        self.depletion_multiplier *= 1.05
                    self.game_phase = "ANAGRAM"
                    self._generate_anagram()
                    # SFX: place_object.wav
                else: # Placement failed
                    reward -= 1.0
                    # SFX: error_buzz.wav
            elif shift_pressed:
                # Penalty: Pressed shift during placement
                reward -= 0.1

        self._update_game_state()
        
        # --- Check Termination ---
        terminated = False
        truncated = False
        if len(self.life_forms) >= self.VICTORY_SCORE:
            reward += 100
            self.game_over = True
            self.game_outcome = "VICTORY"
            terminated = True
        elif self._calculate_total_resources() <= 0:
            reward -= 100
            self.game_over = True
            self.game_outcome = "DEFEAT"
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.game_outcome = "TIMEOUT"
            truncated = True # Use truncated for time limits
            terminated = True

        self.score += reward
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_resources()
        self._render_life_forms()
        if self.game_phase == "PLACEMENT":
            self._render_placement_preview()
        self._render_particles()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ecosystem_size": len(self.life_forms),
            "total_resources": self._calculate_total_resources(),
            "game_phase": self.game_phase,
        }

    # --- Game Logic Helpers ---
    def _generate_anagram(self):
        word = self.np_random.choice(self.word_list)
        scrambled_list = list(word)
        self.np_random.shuffle(scrambled_list)
        scrambled = "".join(scrambled_list)
        while scrambled == word:
            self.np_random.shuffle(scrambled_list)
            scrambled = "".join(scrambled_list)
        self.current_anagram = {"word": word, "scrambled": scrambled}

    def _generate_resource_pools(self):
        self.resource_pools = []
        for _ in range(self.np_random.integers(3, 6)):
            radius = self.np_random.uniform(30, 60)
            self.resource_pools.append({
                "pos": pygame.Vector2(
                    self.np_random.uniform(50, self.WIDTH - 50),
                    self.np_random.uniform(50, self.HEIGHT - 100)
                ),
                "target_radius": radius,
                "current_radius": radius,
            })

    def _generate_life_form_for_placement(self):
        shape = self.np_random.choice(self.LIFE_FORM_SHAPES)
        color_idx = self.np_random.integers(0, len(self.LIFE_FORM_COLORS))
        color = self.LIFE_FORM_COLORS[color_idx]
        size = self.np_random.uniform(12, 18)
        self.current_life_form_to_place = LifeForm(self.cursor_pos, shape, color, size)

    def _move_cursor(self, movement):
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

    def _place_life_form(self):
        form = self.current_life_form_to_place
        if not form: return False, "No form"

        # Check collision with other life forms
        for lf in self.life_forms:
            if self.cursor_pos.distance_to(lf.pos) < lf.get_current_size() + form.get_current_size() + 10:
                return False, "Collision"

        # Check for resources
        nearest_pool_idx = self._find_nearest_pool(self.cursor_pos)
        if nearest_pool_idx == -1: return False, "No resources"
        
        pool = self.resource_pools[nearest_pool_idx]
        placement_cost = form.size * 5
        if pool["current_radius"] < placement_cost: return False, "Insufficient resources"
        
        # --- Placement Success ---
        pool["current_radius"] -= placement_cost
        form.pos.update(self.cursor_pos)
        form.nearest_pool_idx = nearest_pool_idx
        self.life_forms.append(form)
        self.current_life_form_to_place = None

        # Create placement particles
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(20, 41)
            self.particles.append(Particle(self.cursor_pos, vel, form.color, self.COLOR_BG, life))

        return True, "Success"

    def _update_game_state(self):
        # Update life forms and drain resources
        for form in self.life_forms:
            form.update()
            if form.nearest_pool_idx != -1 and form.nearest_pool_idx < len(self.resource_pools):
                pool = self.resource_pools[form.nearest_pool_idx]
                drain_amount = (form.size / 15) * 0.1 * self.depletion_multiplier
                if pool["current_radius"] > 0:
                    pool["current_radius"] = max(0, pool["current_radius"] - drain_amount)
                    
                    # Create resource flow particles
                    if self.steps % 5 == 0:
                        p_start_angle = self.np_random.uniform(0, 360)
                        p_start_radius = self.np_random.uniform(0, pool["current_radius"])
                        p_start = pool["pos"] + pygame.Vector2(p_start_radius, 0).rotate(p_start_angle)
                        vel = (form.pos - p_start).normalize() * 1.5
                        self.particles.append(Particle(p_start, vel, self.COLOR_RESOURCE, self.COLOR_RESOURCE_GLOW, 40))

        # Update particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def _find_nearest_pool(self, pos):
        min_dist = float('inf')
        best_idx = -1
        for i, pool in enumerate(self.resource_pools):
            if pool["current_radius"] > 0:
                dist = pos.distance_to(pool["pos"])
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
        return best_idx
    
    def _calculate_total_resources(self):
        return sum(p['current_radius'] for p in self.resource_pools)

    # --- Rendering Helpers ---
    def _render_background(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_resources(self):
        for pool in self.resource_pools:
            pos = (int(pool["pos"].x), int(pool["pos"].y))
            radius = int(pool["current_radius"])
            if radius > 0:
                self._draw_glowing_circle(self.screen, pos, radius, self.COLOR_RESOURCE, self.COLOR_RESOURCE_GLOW)
    
    def _render_life_forms(self):
        for form in self.life_forms:
            # Draw resource connection line
            if form.nearest_pool_idx != -1 and form.nearest_pool_idx < len(self.resource_pools) and self.resource_pools[form.nearest_pool_idx]['current_radius'] > 0:
                pool = self.resource_pools[form.nearest_pool_idx]
                pool_pos = pool['pos']
                alpha = int(100 * (pool['current_radius'] / pool['target_radius']))
                line_color = (*form.color[:3], alpha)
                pygame.draw.aaline(self.screen, line_color, form.pos, pool_pos, 1)

            # Draw form
            self._draw_polygon(self.screen, form.color, form.pos, form.get_current_size(), form.shape_sides, form.angle)

    def _render_placement_preview(self):
        if not self.current_life_form_to_place: return
        
        # Draw cursor
        cursor_pulse = (math.sin(self.pulse_timer * 2) + 1) / 2
        cursor_color = self.COLOR_CURSOR
        pygame.gfxdraw.aacircle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), int(15 + cursor_pulse * 5), cursor_color)

        # Draw ghost of life form
        form = self.current_life_form_to_place
        ghost_color = (*form.color[:3], 100)
        self._draw_polygon(self.screen, ghost_color, self.cursor_pos, form.get_current_size(), form.shape_sides, form.angle)

        # Draw line to nearest resource
        nearest_pool_idx = self._find_nearest_pool(self.cursor_pos)
        if nearest_pool_idx != -1:
            pool = self.resource_pools[nearest_pool_idx]
            if pool['current_radius'] > form.size * 5:
                line_color = (*self.COLOR_RESOURCE, 150)
            else:
                line_color = (255, 0, 0, 150)
            pygame.draw.aaline(self.screen, line_color, self.cursor_pos, pool['pos'])


    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Panel background
        panel_rect = pygame.Rect(0, self.HEIGHT - 60, self.WIDTH, 60)
        s = pygame.Surface((self.WIDTH, 60), pygame.SRCALPHA)
        s.fill(self.COLOR_PANEL)
        self.screen.blit(s, (0, self.HEIGHT - 60))

        # Ecosystem size
        eco_text = f"ECOSYSTEM: {len(self.life_forms)} / {self.VICTORY_SCORE}"
        self._draw_text(eco_text, (20, 20), font=self.font_medium)
        
        # Resources
        total_res = self._calculate_total_resources()
        max_res = sum(p['target_radius'] for p in self.resource_pools) if self.resource_pools else 1
        res_percent = (total_res / max_res) * 100 if max_res > 0 else 0
        res_text = f"RESOURCES: {res_percent:.1f}%"
        res_color = self.COLOR_TEXT if res_percent > 20 else (255, 80, 80)
        self._draw_text(res_text, (self.WIDTH - 20, 20), font=self.font_medium, color=res_color, align="topright")
        
        # Anagram / Instructions
        if self.game_phase == "ANAGRAM":
            prompt = f"Solve: {self.current_anagram['scrambled']}"
            action_text = "[SHIFT] to Generate New Form"
        else: # PLACEMENT
            prompt = "Select placement for new form"
            action_text = "[ARROWS] to Move, [SPACE] to Place"
        
        self._draw_text(prompt, (self.WIDTH / 2, self.HEIGHT - 45), font=self.font_medium, align="midtop")
        self._draw_text(action_text, (self.WIDTH / 2, self.HEIGHT - 20), font=self.font_small, align="midtop")

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        if self.game_outcome == "VICTORY":
            title_text = "ECOSYSTEM STABLE"
            title_color = (100, 255, 150)
        else:
            title_text = "ECOSYSTEM COLLAPSED"
            title_color = (255, 100, 100)
        
        self._draw_text(title_text, (self.WIDTH/2, self.HEIGHT/2 - 30), font=self.font_large, color=title_color, align="center")
        self._draw_text(f"Final Score: {self.score:.0f}", (self.WIDTH/2, self.HEIGHT/2 + 20), font=self.font_medium, align="center")

    # --- Drawing Primitives ---
    def _draw_polygon(self, surface, color, center, radius, n_sides, rotation=0):
        if n_sides < 3: return
        points = []
        for i in range(n_sides):
            angle = rotation + (2 * math.pi * i / n_sides)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        
        # Glow effect
        glow_color = (*color[:3], 50)
        pygame.gfxdraw.filled_polygon(surface, points, glow_color)
        pygame.gfxdraw.aapolygon(surface, points, glow_color)

        # Main shape
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def _draw_glowing_circle(self, surface, center, radius, color, glow_color):
        radius = int(radius)
        if radius <= 0: return
        for i in range(radius, max(0, radius-10), -1):
            alpha = int(glow_color[3] * (1 - (radius - i) / 10)) if len(glow_color) > 3 else int(50 * (1 - (radius - i) / 10))
            pygame.gfxdraw.aacircle(surface, center[0], center[1], i, (*glow_color[:3], alpha))
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, align="topleft"):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        shadow_pos = (pos[0] + 2, pos[1] + 2)

        if align == "center":
            text_rect.center = pos
            shadow_rect = shadow_surf.get_rect(center=shadow_pos)
        elif align == "topright":
            text_rect.topright = pos
            shadow_rect = shadow_surf.get_rect(topright=shadow_pos)
        elif align == "midtop":
            text_rect.midtop = pos
            shadow_rect = shadow_surf.get_rect(midtop=shadow_pos)
        else: # topleft
            text_rect.topleft = pos
            shadow_rect = shadow_surf.get_rect(topleft=shadow_pos)
            
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and debugging.
    # It will not be run by the evaluation server.
    # To use, you might need to remove the dummy video driver environment variable.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    truncated = False
    
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("GameEnv")

    while running:
        action = [0, 0, 0] # no-op
        
        # For continuous key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}, Truncated: {truncated}")

        # The observation is a numpy array, convert it back to a surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(env.TARGET_FPS)

    env.close()