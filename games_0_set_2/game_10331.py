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

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Defend your central ward from encroaching monsters by matching magical glyphs. "
        "Select and drag glyphs of the same type together to reinforce the ward before it collapses."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to pick up a glyph and "
        "release space over another of the same type to match. Hold shift to cancel a selection."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    TARGET_FPS = 30

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_WARD = (0, 150, 255)
    COLOR_WARD_BOUNDARY = (50, 80, 120)
    COLOR_MONSTER = (100, 20, 40)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_MATCH_SUCCESS = (255, 215, 0)
    COLOR_MATCH_FAIL = (200, 0, 0)
    COLOR_IMPACT = (255, 255, 255)
    GLYPH_COLORS = [
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Yellow
        (0, 255, 0),    # Lime
    ]

    # Game Mechanics
    WARD_MAX_SIZE = 100.0
    WARD_INITIAL_SIZE = 50.0
    WARD_DECAY_RATE = 0.02
    WARD_GROWTH_ON_MATCH = 15.0
    WARD_MAX_RADIUS = 120
    CURSOR_SPEED = 12
    GLYPH_RADIUS = 12
    GLYPH_MATCH_DISTANCE = 30
    MIN_GLYPHS = 6
    MAX_GLYPHS = 10
    MONSTER_DAMAGE = 20.0
    MONSTER_SPAWN_CHANCE_INITIAL = 0.005 # 1/200
    MONSTER_SPEED = 1.0

    # Rewards
    REWARD_MATCH = 0.1
    REWARD_WIN = 100.0
    REWARD_LOSS = -100.0

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_ward = pygame.font.SysFont("Consolas", 32, bold=True)
        
        self.center_x = self.SCREEN_WIDTH // 2
        self.center_y = self.SCREEN_HEIGHT // 2

        self.steps = 0
        self.score = 0.0
        self.ward_size = 0.0
        self.time_scale = 1.0
        self.monster_spawn_chance = 0.0
        self.cursor_pos = [0, 0]
        self.selected_glyph_idx = None
        self.prev_space_held = False
        self.glyphs = []
        self.monsters = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.ward_size = self.WARD_INITIAL_SIZE
        self.time_scale = 1.0
        self.monster_spawn_chance = self.MONSTER_SPAWN_CHANCE_INITIAL
        
        self.cursor_pos = [self.center_x, self.center_y]
        self.selected_glyph_idx = None
        self.prev_space_held = False

        self.glyphs = []
        self.monsters = []
        self.particles = []

        for _ in range(self.MAX_GLYPHS):
            self._spawn_glyph()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        self._move_cursor(movement)
        match_reward = self._handle_input(space_held, shift_held)
        reward += match_reward
        self.score += match_reward

        self._update_monsters()
        self._update_ward()
        self._update_particles()
        self._spawn_new_entities()

        self.steps += 1
        if self.steps % 100 == 0 and self.steps > 0:
            self.time_scale += 0.001
            self.monster_spawn_chance += 0.0005

        terminated = self.ward_size <= 0 or self.steps >= self.MAX_STEPS
        truncated = False # This game does not have a truncation condition separate from termination
        
        if terminated:
            if self.ward_size <= 0:
                reward += self.REWARD_LOSS
            elif self.steps >= self.MAX_STEPS:
                reward += self.REWARD_WIN

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _move_cursor(self, movement):
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

    def _handle_input(self, space_held, shift_held):
        reward = 0
        space_pressed = space_held and not self.prev_space_held
        space_released = not space_held and self.prev_space_held
        self.prev_space_held = space_held

        if shift_held and self.selected_glyph_idx is not None:
            self.selected_glyph_idx = None

        if space_pressed and self.selected_glyph_idx is None:
            for i, glyph in enumerate(self.glyphs):
                dist = math.hypot(self.cursor_pos[0] - glyph['pos'][0], self.cursor_pos[1] - glyph['pos'][1])
                if dist < self.GLYPH_RADIUS:
                    self.selected_glyph_idx = i
                    break
        
        if space_released and self.selected_glyph_idx is not None:
            selected = self.glyphs[self.selected_glyph_idx]
            target_idx = None
            for i, glyph in enumerate(self.glyphs):
                if i == self.selected_glyph_idx: continue
                dist = math.hypot(self.cursor_pos[0] - glyph['pos'][0], self.cursor_pos[1] - glyph['pos'][1])
                if dist < self.GLYPH_MATCH_DISTANCE:
                    target_idx = i
                    break
            
            if target_idx is not None and self.glyphs[target_idx]['type'] == selected['type']:
                # Successful Match
                reward = self.REWARD_MATCH
                self.ward_size = min(self.WARD_MAX_SIZE, self.ward_size + self.WARD_GROWTH_ON_MATCH)
                
                match_pos = self.glyphs[target_idx]['pos']
                self._create_particles(match_pos, 30, self.COLOR_MATCH_SUCCESS, (1, 4), (20, 40))
                
                indices = sorted([self.selected_glyph_idx, target_idx], reverse=True)
                for i in indices:
                    self.glyphs.pop(i)
            else:
                if target_idx is not None: # Mismatched glyphs
                    self._create_particles(self.cursor_pos, 10, self.COLOR_MATCH_FAIL, (0.5, 2), (10, 20))
            self.selected_glyph_idx = None

        return reward

    def _update_ward(self):
        self.ward_size -= self.WARD_DECAY_RATE * self.time_scale
        self.ward_size = max(0, self.ward_size)

    def _update_monsters(self):
        current_ward_radius = (self.ward_size / self.WARD_MAX_SIZE) * self.WARD_MAX_RADIUS
        for monster in self.monsters[:]:
            angle = math.atan2(self.center_y - monster['pos'][1], self.center_x - monster['pos'][0])
            speed = self.MONSTER_SPEED * self.time_scale
            monster['pos'][0] += math.cos(angle) * speed
            monster['pos'][1] += math.sin(angle) * speed
            
            dist_to_center = math.hypot(monster['pos'][0] - self.center_x, monster['pos'][1] - self.center_y)
            if dist_to_center < current_ward_radius:
                self.ward_size -= self.MONSTER_DAMAGE
                self._create_particles(monster['pos'], 20, self.COLOR_IMPACT, (1, 3), (15, 30))
                self.monsters.remove(monster)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_new_entities(self):
        if len(self.glyphs) < self.MIN_GLYPHS:
            self._spawn_glyph()
        if self.np_random.random() < self.monster_spawn_chance * self.time_scale:
            self._spawn_monster()

    def _spawn_glyph(self):
        if len(self.glyphs) >= self.MAX_GLYPHS:
            return
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        dist = self.np_random.uniform(self.GLYPH_RADIUS * 2, self.WARD_MAX_RADIUS - self.GLYPH_RADIUS)
        pos = [
            self.center_x + math.cos(angle) * dist,
            self.center_y + math.sin(angle) * dist,
        ]
        glyph_type = self.np_random.integers(0, len(self.GLYPH_COLORS))
        self.glyphs.append({'pos': pos, 'type': glyph_type})

    def _spawn_monster(self):
        edge = self.np_random.integers(0, 4)
        if edge == 0: # Top
            pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), -20]
        elif edge == 1: # Bottom
            pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20]
        elif edge == 2: # Left
            pos = [-20, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
        else: # Right
            pos = [self.SCREEN_WIDTH + 20, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
        self.monsters.append({'pos': pos})

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
            "ward_size": self.ward_size,
            "time_scale": self.time_scale,
        }

    def _render_game(self):
        self._render_background_art()
        self._render_ward()
        self._render_glyphs()
        self._render_monsters()
        self._render_particles()
        self._render_cursor()

    def _render_background_art(self):
        for i in range(5):
            radius = int(self.WARD_MAX_RADIUS * (1.2 + i * 0.2) + 10 * math.sin(self.steps * 0.01 + i))
            alpha = 20 - i * 3
            pygame.gfxdraw.aacircle(self.screen, self.center_x, self.center_y, radius, (*self.COLOR_WARD_BOUNDARY, alpha))

    def _render_ward(self):
        pulse = 5 * math.sin(self.steps * 0.1)
        current_radius = int((self.ward_size / self.WARD_MAX_SIZE) * self.WARD_MAX_RADIUS)
        
        pygame.gfxdraw.aacircle(self.screen, self.center_x, self.center_y, self.WARD_MAX_RADIUS, self.COLOR_WARD_BOUNDARY)
        
        if current_radius > 0:
            ward_color = self.COLOR_WARD
            alpha = int(100 + 50 * (self.ward_size / self.WARD_MAX_SIZE))
            
            for i in range(4):
                glow_radius = int(current_radius + pulse + i * 2)
                glow_alpha = int(alpha * 0.2 * (1 - i/4))
                if glow_radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, self.center_x, self.center_y, glow_radius, (*ward_color, glow_alpha))

            pygame.gfxdraw.filled_circle(self.screen, self.center_x, self.center_y, int(current_radius + pulse), (*ward_color, alpha))
            pygame.gfxdraw.aacircle(self.screen, self.center_x, self.center_y, int(current_radius + pulse), (*ward_color, 200))

    def _render_glyphs(self):
        for i, glyph in enumerate(self.glyphs):
            pos = self.cursor_pos if i == self.selected_glyph_idx else glyph['pos']
            self._draw_glyph(self.screen, glyph['type'], pos, self.GLYPH_RADIUS)

    def _draw_glyph(self, surface, glyph_type, pos, radius):
        color = self.GLYPH_COLORS[glyph_type]
        glow_color = (*color, 50)
        x, y = int(pos[0]), int(pos[1])
        
        pygame.gfxdraw.filled_circle(surface, x, y, int(radius * 1.5), glow_color)
        pygame.gfxdraw.filled_circle(surface, x, y, radius, self.COLOR_BG)

        if glyph_type == 0: # Circle
            pygame.gfxdraw.aacircle(surface, x, y, int(radius * 0.6), color)
            pygame.gfxdraw.filled_circle(surface, x, y, int(radius * 0.6), color)
        elif glyph_type == 1: # Square
            r = int(radius * 0.6)
            pygame.draw.rect(surface, color, (x - r, y - r, 2*r, 2*r), 0)
        elif glyph_type == 2: # Triangle
            r = int(radius * 0.7)
            points = [(x, y - r), (x - r, y + r//2), (x + r, y + r//2)]
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
        elif glyph_type == 3: # Cross
            r = int(radius * 0.7)
            pygame.draw.line(surface, color, (x - r, y), (x + r, y), 3)
            pygame.draw.line(surface, color, (x, y - r), (x, y + r), 3)
        
        pygame.gfxdraw.aacircle(surface, x, y, radius, color)

    def _render_monsters(self):
        for monster in self.monsters:
            x, y = int(monster['pos'][0]), int(monster['pos'][1])
            for i in range(5):
                offset_x = self.np_random.uniform(-5, 5)
                offset_y = self.np_random.uniform(-5, 5)
                radius = self.np_random.uniform(8, 15)
                alpha = self.np_random.uniform(50, 100)
                pygame.gfxdraw.filled_circle(self.screen, int(x+offset_x), int(y+offset_y), int(radius), (*self.COLOR_MONSTER, int(alpha)))

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], int(alpha))
            radius = int(p['radius'] * (p['life'] / p['max_life']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)

    def _render_cursor(self):
        x, y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        length = 8
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x - length, y), (x + length, y), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x, y - length), (x, y + length), 2)
        
        if self.selected_glyph_idx is not None:
            pygame.gfxdraw.aacircle(self.screen, x, y, 12, self.COLOR_CURSOR)

    def _render_ui(self):
        time_text = f"TIME: {self.steps / self.TARGET_FPS:.1f}s"
        time_surf = self.font_main.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        ward_text = f"{int(self.ward_size)}%"
        ward_surf = self.font_ward.render(ward_text, True, self.COLOR_TEXT)
        text_rect = ward_surf.get_rect(center=(self.center_x, self.center_y))
        self.screen.blit(ward_surf, text_rect)

    def _create_particles(self, pos, count, color, speed_range, life_range):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(*speed_range)
            life = self.np_random.uniform(*life_range)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block is for human play and debugging purposes.
    # It is not used by the evaluation environment.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a visible driver
    env = GameEnv()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Glyph Ward")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    while not done:
        movement = 0 # none
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        if info['steps'] > 0 and info['steps'] % 10 == 0:
            print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Ward: {info['ward_size']:.1f}%")

        clock.tick(env.TARGET_FPS)

    print(f"Game Over! Final Score: {info['score']:.2f} in {info['steps']} steps.")
    env.close()