
# Generated: 2025-08-28T02:04:35.745596
# Source Brief: brief_04324.md
# Brief Index: 4324

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to slice vertically. "
        "Slice fruit, avoid bombs!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game where you slice falling fruit to score points while avoiding bombs. "
        "Precise timing is key!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_TOP = (40, 40, 60)
    COLOR_BG_BOTTOM = (20, 20, 40)
    COLOR_SLICER_CURSOR = (255, 255, 255)
    COLOR_SLICE_EFFECT = (255, 255, 255)
    COLOR_BOMB = (50, 50, 50)
    COLOR_BOMB_FLASH = (255, 50, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SCORE_POPUP = (255, 220, 100)
    COLOR_SHADOW = (0, 0, 0, 100)

    FRUIT_TYPES = {
        "apple": {"color": (255, 60, 60), "points": 1, "radius": 15},
        "orange": {"color": (255, 165, 0), "points": 2, "radius": 16},
        "lemon": {"color": (255, 255, 100), "points": 3, "radius": 14},
        "kiwi": {"color": (100, 200, 80), "points": 1, "radius": 12},
    }

    # Game parameters
    WIN_SCORE = 100
    MAX_BOMBS_HIT = 3
    MAX_STEPS = 1500
    SLICER_SPEED = 15
    BOMB_RADIUS = 18
    BONUS_DISTANCE = 70

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_popup = pygame.font.Font(None, 28)
        self.font_gameover = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.slicer_pos = [0, 0]
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.slice_effects = []
        self.score_popups = []
        self.steps = 0
        self.score = 0
        self.bombs_hit = 0
        self.game_over = False
        self.prev_space_held = False
        self.spawn_timer = 0
        self.base_speed = 0.0
        self.spawn_interval = 0
        self.screen_shake = 0
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.slicer_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2]
        self.fruits.clear()
        self.bombs.clear()
        self.particles.clear()
        self.slice_effects.clear()
        self.score_popups.clear()
        
        self.steps = 0
        self.score = 0
        self.bombs_hit = 0
        self.game_over = False
        self.prev_space_held = False
        
        self.spawn_timer = 0
        self.base_speed = 1.5
        self.spawn_interval = 45 # Spawn every 45 frames initially
        self.screen_shake = 0
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # If game is over, do nothing and return terminal state
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        self.steps += 1
        step_reward = 0.0

        # Update slicer and handle slice action
        self._update_slicer_pos(movement)
        slice_triggered = space_held and not self.prev_space_held
        if slice_triggered:
            # SFX: whoosh.wav
            reward_from_slice = self._handle_slice()
            step_reward += reward_from_slice
        self.prev_space_held = space_held

        # Update world state (spawns, physics, effects)
        self._update_difficulty()
        self._update_spawns()
        self._update_objects_and_effects()

        # Check for termination
        terminated = (
            self.score >= self.WIN_SCORE
            or self.bombs_hit >= self.MAX_BOMBS_HIT
            or self.steps >= self.MAX_STEPS
        )
        
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                step_reward += 100.0
            elif self.bombs_hit >= self.MAX_BOMBS_HIT:
                step_reward -= 100.0
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_slicer_pos(self, movement):
        if movement == 1:  # Up
            self.slicer_pos[1] -= self.SLICER_SPEED
        elif movement == 2:  # Down
            self.slicer_pos[1] += self.SLICER_SPEED
        elif movement == 3:  # Left
            self.slicer_pos[0] -= self.SLICER_SPEED
        elif movement == 4:  # Right
            self.slicer_pos[0] += self.SLICER_SPEED
        
        self.slicer_pos[0] = np.clip(self.slicer_pos[0], 0, self.SCREEN_WIDTH)
        self.slicer_pos[1] = np.clip(self.slicer_pos[1], 0, self.SCREEN_HEIGHT)

    def _handle_slice(self):
        slice_x = self.slicer_pos[0]
        self.slice_effects.append({"x": slice_x, "life": 10})
        
        reward = 0.0
        
        # Check for bomb slices first
        for bomb in self.bombs[:]:
            if abs(bomb['pos'][0] - slice_x) < self.BOMB_RADIUS:
                self.bombs.remove(bomb)
                self.bombs_hit += 1
                reward -= 5.0
                self._create_explosion(bomb['pos'])
                self.screen_shake = 10
                # SFX: explosion.wav
        
        # Check for fruit slices
        for fruit in self.fruits[:]:
            if abs(fruit['pos'][0] - slice_x) < fruit['radius']:
                self.fruits.remove(fruit)
                
                points = fruit['points']
                self.score += points
                reward += 1.0
                
                is_bonus = False
                for bomb in self.bombs:
                    dist = math.hypot(fruit['pos'][0] - bomb['pos'][0], fruit['pos'][1] - bomb['pos'][1])
                    if dist < self.BONUS_DISTANCE + self.BOMB_RADIUS:
                        is_bonus = True
                        break
                
                if is_bonus:
                    reward += 5.0
                    points_str = f"+{points} BONUS!"
                else:
                    points_str = f"+{points}"
                
                self.score_popups.append({"pos": list(fruit['pos']), "text": points_str, "life": 45, "color": self.COLOR_SCORE_POPUP})
                self._create_fruit_particles(fruit['pos'], fruit['color'])
                # SFX: slice.wav
                
        return reward

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 100 == 0:
            self.base_speed += 0.2
            self.spawn_interval = max(10, int(self.spawn_interval - 2))

    def _update_spawns(self):
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self.spawn_timer = self.spawn_interval
            
            spawn_x = self.np_random.uniform(50, self.SCREEN_WIDTH - 50)
            speed_y = self.base_speed + self.np_random.uniform(0.5, 2.0)
            speed_x = self.np_random.uniform(-1.0, 1.0)
            
            # Bomb probability increases over time
            bomb_chance = 0.25 + (self.steps / self.MAX_STEPS) * 0.2
            if self.np_random.random() < bomb_chance:
                self.bombs.append({"pos": [spawn_x, -self.BOMB_RADIUS], "vel": [speed_x, speed_y]})
            else:
                fruit_name = self.np_random.choice(list(self.FRUIT_TYPES.keys()))
                fruit_info = self.FRUIT_TYPES[fruit_name]
                self.fruits.append({
                    "pos": [spawn_x, -fruit_info['radius']],
                    "vel": [speed_x, speed_y],
                    "radius": fruit_info['radius'],
                    "color": fruit_info['color'],
                    "points": fruit_info['points'],
                })

    def _update_objects_and_effects(self):
        for obj_list in [self.fruits, self.bombs]:
            for obj in obj_list[:]:
                obj['pos'][0] += obj['vel'][0]
                obj['pos'][1] += obj['vel'][1]
                radius = obj.get('radius', self.BOMB_RADIUS)
                if obj['pos'][1] > self.SCREEN_HEIGHT + radius:
                    obj_list.remove(obj)

        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        for s in self.slice_effects[:]:
            s['life'] -= 1
            if s['life'] <= 0:
                self.slice_effects.remove(s)

        for pop in self.score_popups[:]:
            pop['pos'][1] -= 0.5
            pop['life'] -= 1
            if pop['life'] <= 0:
                self.score_popups.remove(pop)
        
        if self.screen_shake > 0:
            self.screen_shake -= 1

    def _create_explosion(self, pos):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 8)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(30, 60),
                "color": self.np_random.choice([(255, 50, 50), (255, 150, 0), (100, 100, 100)]),
                "size": self.np_random.uniform(2, 5)
            })

    def _create_fruit_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(20, 40),
                "color": color,
                "size": self.np_random.uniform(2, 4)
            })

    def _get_observation(self):
        render_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self._render_background(render_surface)
        self._render_effects(render_surface)
        self._render_objects(render_surface)
        self._render_slicer(render_surface)
        self._render_ui(render_surface)

        if self.screen_shake > 0:
            shake_x = self.np_random.integers(-self.screen_shake, self.screen_shake)
            shake_y = self.np_random.integers(-self.screen_shake, self.screen_shake)
            self.screen.fill((0,0,0))
            self.screen.blit(render_surface, (shake_x, shake_y))
        else:
            self.screen.blit(render_surface, (0,0))
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, surface):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = tuple(int(c1 * (1 - interp) + c2 * interp) for c1, c2 in zip(self.COLOR_BG_TOP, self.COLOR_BG_BOTTOM))
            pygame.draw.line(surface, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_effects(self, surface):
        temp_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        for p in self.particles:
            alpha = int(255 * (p['life'] / 60))
            if alpha > 0:
                color = (*p['color'], alpha)
                pygame.draw.rect(temp_surf, color, (int(p['pos'][0]), int(p['pos'][1]), max(1, int(p['size'])), max(1, int(p['size']))))
        
        for s in self.slice_effects:
            alpha = int(255 * (s['life'] / 10))
            if alpha > 0:
                pygame.draw.rect(temp_surf, (*self.COLOR_SLICE_EFFECT, alpha), (int(s['x'] - 2), 0, 4, self.SCREEN_HEIGHT))
        surface.blit(temp_surf, (0, 0))
    
    def _render_objects(self, surface):
        # Render shadows first
        shadow_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        for obj in self.fruits + self.bombs:
            radius = obj.get('radius', self.BOMB_RADIUS)
            pygame.gfxdraw.filled_circle(shadow_surface, int(obj['pos'][0] + 3), int(obj['pos'][1] + 3), radius, self.COLOR_SHADOW)
        surface.blit(shadow_surface, (0, 0))

        # Render objects
        for bomb in self.bombs:
            pos_int = (int(bomb['pos'][0]), int(bomb['pos'][1]))
            pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], self.BOMB_RADIUS, self.COLOR_BOMB)
            pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], self.BOMB_RADIUS, self.COLOR_BOMB)
            flash_alpha = 128 + 127 * math.sin(self.steps * 0.2)
            flash_color = (*self.COLOR_BOMB_FLASH, flash_alpha)
            pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], 5, flash_color)

        for fruit in self.fruits:
            pos_int = (int(fruit['pos'][0]), int(fruit['pos'][1]))
            pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], fruit['radius'], fruit['color'])
            pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], fruit['radius'], fruit['color'])
            shine_pos = (pos_int[0] - fruit['radius']//2, pos_int[1] - fruit['radius']//2)
            pygame.gfxdraw.filled_circle(surface, shine_pos[0], shine_pos[1], 3, (255, 255, 255, 100))

    def _render_slicer(self, surface):
        x, y = int(self.slicer_pos[0]), int(self.slicer_pos[1])
        pygame.draw.line(surface, self.COLOR_SLICER_CURSOR, (x - 10, y), (x + 10, y), 2)
        pygame.draw.line(surface, self.COLOR_SLICER_CURSOR, (x, y - 10), (x, y + 10), 2)
        pygame.gfxdraw.aacircle(surface, x, y, 8, self.COLOR_SLICER_CURSOR)

    def _render_ui(self, surface):
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        surface.blit(score_text, (10, 10))

        for i in range(self.MAX_BOMBS_HIT):
            pos = (self.SCREEN_WIDTH - 40 - i * 35, 15)
            if i < self.MAX_BOMBS_HIT - self.bombs_hit:
                pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], 12, self.COLOR_BOMB)
            else:
                pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], 12, (50, 50, 50, 80))
                pygame.draw.line(surface, (255,50,50, 150), (pos[0]-8, pos[1]-8), (pos[0]+8, pos[1]+8), 3)
                pygame.draw.line(surface, (255,50,50, 150), (pos[0]-8, pos[1]+8), (pos[0]+8, pos[1]-8), 3)

        popup_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        for pop in self.score_popups:
            alpha = int(255 * (pop['life'] / 45))
            if alpha > 0:
                text_surf = self.font_popup.render(pop['text'], True, pop['color'])
                text_surf.set_alpha(alpha)
                popup_surface.blit(text_surf, (int(pop['pos'][0]), int(pop['pos'][1])))
        surface.blit(popup_surface, (0, 0))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = (100, 255, 100) if self.score >= self.WIN_SCORE else (255, 100, 100)
            text_surf = self.font_gameover.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            surface.blit(overlay, (0, 0))
            surface.blit(text_surf, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "bombs_hit": self.bombs_hit}

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    import os
    # Set the SDL_VIDEODRIVER for your specific environment if needed
    # 'x11', 'dummydriver', 'windows' etc.
    # os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Fruit Slicer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    while running:
        movement, space_held, shift_held = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Transpose for pygame display (from H,W,C to W,H,C)
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0.0
        
        if terminated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            # Wait for user to press 'R' to reset
            pass

        clock.tick(30)
        
    env.close()