
# Generated: 2025-08-27T14:20:49.036995
# Source Brief: brief_00650.md
# Brief Index: 650

        
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
        "Controls: Arrow keys to move your slicer. Hold space to slice. "
        "Slice the fruit, but avoid the bombs!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice falling fruit while dodging bombs in this fast-paced arcade game. "
        "Reach 50 points to win, but hitting 3 bombs means game over."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAYER_SPEED = 10
    MAX_STEPS = 1000
    WIN_SCORE = 50
    MAX_BOMBS_HIT = 3
    
    # Colors
    COLOR_BG_TOP = (20, 30, 40)
    COLOR_BG_BOTTOM = (40, 60, 80)
    COLOR_PLAYER = (255, 255, 100)
    COLOR_TRAIL = (255, 255, 200)
    COLOR_BOMB = (30, 30, 30)
    COLOR_FUSE = (150, 80, 0)
    COLOR_SPARK = (255, 200, 0)
    COLOR_FRUIT_RED = (220, 50, 50)
    COLOR_FRUIT_GREEN = (50, 220, 50)
    COLOR_FRUIT_ORANGE = (255, 150, 20)
    COLOR_FRUIT_GOLD = (255, 215, 0)
    COLOR_SHINE = (255, 255, 255, 100)
    COLOR_TEXT = (255, 255, 255)
    COLOR_BOMB_ICON_OFF = (80, 80, 80)
    COLOR_BOMB_ICON_ON = (255, 0, 0)
    
    FRUIT_COLORS = [COLOR_FRUIT_RED, COLOR_FRUIT_GREEN, COLOR_FRUIT_ORANGE]

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
        self.font_large = pygame.font.Font(None, 72)
        self.font_small = pygame.font.Font(None, 36)
        
        self.np_random = None
        self.game_objects = []
        self.particles = []
        self.slice_trail = []

        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.bombs_hit = 0
        self.game_over = False
        self.win = False

        self.player_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2]
        self.is_slicing = False
        
        self.game_objects = []
        self.particles = []
        self.slice_trail = []

        self.fall_speed = 2.0
        self.fruit_spawn_counter = 0
        self.object_spawn_timer = 0
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1

        # --- 1. Unpack Action and Update Player ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        self._update_player(movement, space_held)
        
        # --- 2. Update Game Logic ---
        self._update_game_state()
        
        # --- 3. Handle Collisions and Assign Rewards ---
        reward += self._handle_collisions()

        # --- 4. Spawn New Objects ---
        self._spawn_objects()

        # --- 5. Check Termination Conditions ---
        if self.score >= self.WIN_SCORE:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100
        elif self.bombs_hit >= self.MAX_BOMBS_HIT:
            self.game_over = True
            terminated = True
            reward -= 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_player(self, movement, space_held):
        if movement == 1:  # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED

        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.SCREEN_WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.SCREEN_HEIGHT)
        
        self.is_slicing = space_held
        if self.is_slicing:
            self.slice_trail.append(list(self.player_pos))
            if len(self.slice_trail) > 10:
                self.slice_trail.pop(0)
        else:
            self.slice_trail.clear()

    def _update_game_state(self):
        # Update fall speed
        if self.steps > 0 and self.steps % 50 == 0:
            self.fall_speed += 0.05
        
        # Update game objects
        for obj in self.game_objects[:]:
            obj['pos'][1] += self.fall_speed
            obj['angle'] = (obj['angle'] + obj['rot_speed']) % 360
            if obj['pos'][1] > self.SCREEN_HEIGHT + obj['radius']:
                self.game_objects.remove(obj)

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        step_reward = 0
        if not self.is_slicing:
            return step_reward
            
        for obj in self.game_objects[:]:
            dist = math.hypot(obj['pos'][0] - self.player_pos[0], obj['pos'][1] - self.player_pos[1])
            if dist < obj['radius'] + 10: # 10 is slicer radius
                if obj['type'] == 'bomb':
                    self.bombs_hit += 1
                    step_reward -= 5
                    self._create_explosion(obj['pos'])
                    # SFX: Explosion
                else: # Fruit
                    if obj['type'] == 'golden_fruit':
                        self.score += 2
                        step_reward += 2
                    else:
                        self.score += 1
                        step_reward += 1
                    self._create_fruit_splash(obj['pos'], obj['color'])
                    # SFX: Juicy slice
                
                self.game_objects.remove(obj)
        return step_reward

    def _spawn_objects(self):
        self.object_spawn_timer -= 1
        if self.object_spawn_timer <= 0:
            self.object_spawn_timer = self.np_random.integers(15, 30)
            
            spawn_x = self.np_random.uniform(50, self.SCREEN_WIDTH - 50)
            radius = self.np_random.uniform(20, 35)
            rot_speed = self.np_random.uniform(-2, 2)
            
            if self.np_random.random() < 0.2: # 20% chance of bomb
                obj = {
                    'type': 'bomb', 'pos': [spawn_x, -radius], 'radius': 30, 'color': self.COLOR_BOMB,
                    'angle': 0, 'rot_speed': 0
                }
            else: # 80% chance of fruit
                self.fruit_spawn_counter += 1
                if self.fruit_spawn_counter % 20 == 0:
                    obj = {
                        'type': 'golden_fruit', 'pos': [spawn_x, -radius], 'radius': radius,
                        'color': self.COLOR_FRUIT_GOLD, 'angle': 0, 'rot_speed': rot_speed
                    }
                else:
                    obj = {
                        'type': 'fruit', 'pos': [spawn_x, -radius], 'radius': radius,
                        'color': random.choice(self.FRUIT_COLORS), 'angle': 0, 'rot_speed': rot_speed
                    }
            self.game_objects.append(obj)

    def _create_fruit_splash(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(20, 40),
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _create_explosion(self, pos):
        # Big flash
        self.particles.append({
            'pos': list(pos), 'vel': [0, 0], 'life': 8,
            'color': (255, 255, 255), 'radius': 100, 'type': 'flash'
        })
        # Fire particles
        for _ in range(40):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 8)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(30, 60),
                'color': random.choice([(255,100,0), (255,200,0), (200,50,0)]),
                'radius': self.np_random.uniform(3, 7)
            })

    def _get_observation(self):
        self._render_background()
        self._render_objects()
        self._render_particles()
        self._render_slicer()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_objects(self):
        for obj in sorted(self.game_objects, key=lambda o: o['radius'], reverse=True):
            x, y = int(obj['pos'][0]), int(obj['pos'][1])
            radius = int(obj['radius'])
            
            if obj['type'] == 'bomb':
                pygame.gfxdraw.filled_circle(self.screen, x, y, radius, self.COLOR_BOMB)
                pygame.gfxdraw.aacircle(self.screen, x, y, radius, self.COLOR_BOMB)
                # Fuse
                fuse_end_x = x + 8
                fuse_end_y = y - radius - 5
                pygame.draw.rect(self.screen, self.COLOR_FUSE, (x, y - radius - 5, 8, 5))
                # Spark
                if self.steps % 4 < 2:
                    spark_radius = self.np_random.integers(2, 4)
                    pygame.gfxdraw.filled_circle(self.screen, fuse_end_x, fuse_end_y, spark_radius, self.COLOR_SPARK)
            else:
                # Fruit body
                pygame.gfxdraw.filled_circle(self.screen, x, y, radius, obj['color'])
                pygame.gfxdraw.aacircle(self.screen, x, y, radius, tuple(c*0.8 for c in obj['color']))
                # Shine
                shine_rect = pygame.Rect(0, 0, radius, radius)
                shine_rect.center = (x - radius * 0.2, y - radius * 0.2)
                shine_surf = pygame.Surface(shine_rect.size, pygame.SRCALPHA)
                pygame.draw.arc(shine_surf, self.COLOR_SHINE, (0, 0, radius, radius), 0.8, 1.8, int(radius/4))
                self.screen.blit(shine_surf, shine_rect)
                # Golden glow
                if obj['type'] == 'golden_fruit':
                    glow_radius = int(radius * 1.5)
                    glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                    alpha = 100 + 30 * math.sin(self.steps * 0.1)
                    pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (255, 220, 50, alpha))
                    self.screen.blit(glow_surf, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)


    def _render_particles(self):
        for p in self.particles:
            x, y = int(p['pos'][0]), int(p['pos'][1])
            if p.get('type') == 'flash':
                alpha = max(0, 255 * (p['life'] / 8))
                glow_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (p['color'][0], p['color'][1], p['color'][2], alpha), (p['radius'], p['radius']), p['radius'])
                self.screen.blit(glow_surf, (x - p['radius'], y - p['radius']), special_flags=pygame.BLEND_RGBA_ADD)
            else:
                alpha = max(0, 255 * (p['life'] / 40))
                color_with_alpha = (p['color'][0], p['color'][1], p['color'][2], alpha)
                pygame.gfxdraw.filled_circle(self.screen, x, y, int(p['radius']), color_with_alpha)
    
    def _render_slicer(self):
        # Trail
        if len(self.slice_trail) > 1:
            for i in range(len(self.slice_trail) - 1):
                start_pos = self.slice_trail[i]
                end_pos = self.slice_trail[i+1]
                alpha = int(255 * (i / len(self.slice_trail)))
                width = int(5 + 15 * (i / len(self.slice_trail)))
                pygame.draw.line(self.screen, self.COLOR_TRAIL + (alpha,), start_pos, end_pos, width)
        
        # Cursor
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        radius = 10
        if self.is_slicing:
            # Pulsing effect when slicing
            pulse_radius = radius * (1.2 + 0.2 * math.sin(self.steps * 0.5))
            glow_surf = pygame.Surface((pulse_radius * 2, pulse_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, int(pulse_radius), int(pulse_radius), int(pulse_radius), (255, 255, 255, 80))
            self.screen.blit(glow_surf, (px - pulse_radius, py - pulse_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Bomb strikes
        for i in range(self.MAX_BOMBS_HIT):
            color = self.COLOR_BOMB_ICON_ON if i < self.bombs_hit else self.COLOR_BOMB_ICON_OFF
            pygame.gfxdraw.filled_circle(self.screen, self.SCREEN_WIDTH - 25 - (i * 35), 25, 12, color)
            pygame.gfxdraw.aacircle(self.screen, self.SCREEN_WIDTH - 25 - (i * 35), 25, 12, color)

        # Game Over / Win Text
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Shadow
            shadow_surf = self.font_large.render(msg, True, (0,0,0,150))
            self.screen.blit(shadow_surf, text_rect.move(3,3))
            
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bombs_hit": self.bombs_hit,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    import os
    # Set a video driver that works for your system.
    # Common options: "x11", "windows", "dummy", "quartz" (for macOS).
    # If you are running in a headless environment, "dummy" is a good choice.
    if "SDL_VIDEODRIVER" not in os.environ:
         os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv()
    obs, info = env.reset()
    
    # The display is only needed for human play
    if os.environ["SDL_VIDEODRIVER"] != "dummy":
        display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Fruit Slicer")
    else:
        display_screen = None

    terminated = False
    clock = pygame.time.Clock()
    
    print(GameEnv.user_guide)
    
    while not terminated:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        # Human controls
        if display_screen:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
        else: # Simple agent for dummy mode
            action = env.action_space.sample()
        
        if display_screen:
            action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if display_screen:
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        clock.tick(30) # Match the intended FPS
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            if display_screen:
                # Wait a bit before closing
                pygame.time.wait(2000)

    env.close()