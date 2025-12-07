
# Generated: 2025-08-27T16:53:35.352743
# Source Brief: brief_01364.md
# Brief Index: 1364

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    user_guide = (
        "Controls: Arrow keys to jump between tiles. Avoid the encroaching shadows. "
        "Collect glowing fragments to score points and escape through the green exit."
    )

    game_description = (
        "An isometric survival game. You are a being of light trapped in a shadowy world. "
        "Jump between platforms to evade the pulsing, dangerous shadows and collect light fragments. "
        "Reach the exit to escape the darkness."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (15, 18, 28)
    COLOR_GRID = (30, 35, 50)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_SHADOW = (0, 0, 0, 100)
    COLOR_SHADOW = (5, 5, 10)
    COLOR_ITEM = (255, 220, 50)
    COLOR_EXIT = (50, 200, 100)
    COLOR_TEXT = (230, 230, 240)
    COLOR_PARTICLE_ITEM = (255, 230, 100)
    COLOR_PARTICLE_DEATH = (200, 50, 50)

    # Grid and Isometric Projection
    GRID_WIDTH = 11
    GRID_HEIGHT = 11
    TILE_WIDTH = 48
    TILE_HEIGHT = 24
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 80

    # Game Mechanics
    JUMP_DURATION = 15  # frames/steps for a jump to complete
    JUMP_HEIGHT = 40
    INITIAL_SHADOW_SPEED = 0.08
    SHADOW_SPEED_INCREASE_INTERVAL = 200
    SHADOW_SPEED_INCREASE_AMOUNT = 0.05

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # This will be set properly in reset
        self.np_random = None

        # Initialize empty state variables
        self.player_pos = [0, 0]
        self.jump_state = {'is_jumping': False, 'progress': 0, 'start_pos': [0,0], 'target_pos': [0,0]}
        self.shadows = []
        self.items = []
        self.exit_pos = [0, 0]
        self.particles = []
        self.steps = 0
        self.score = 0
        self.shadow_speed = self.INITIAL_SHADOW_SPEED
        self.game_over_message = ""
        
        # This call is for development and self-checking
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over_message = ""
        self.shadow_speed = self.INITIAL_SHADOW_SPEED
        self.particles.clear()

        # Place player
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT - 1]
        self.jump_state = {'is_jumping': False, 'progress': 0, 'start_pos': self.player_pos, 'target_pos': self.player_pos}

        # Place exit
        self.exit_pos = [self.GRID_WIDTH // 2, 0]

        # Place items and shadows
        self.items.clear()
        self.shadows.clear()
        
        occupied_tiles = {tuple(self.player_pos), tuple(self.exit_pos)}
        
        # Place 3 items
        for _ in range(3):
            while True:
                pos = [self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(1, self.GRID_HEIGHT - 1)]
                if tuple(pos) not in occupied_tiles:
                    self.items.append({'pos': pos, 'pulse': self.np_random.random() * math.pi * 2})
                    occupied_tiles.add(tuple(pos))
                    break
        
        # Place 5 shadows
        for _ in range(5):
             while True:
                pos = [self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)]
                if tuple(pos) not in occupied_tiles:
                    self.shadows.append({
                        'pos': pos, 
                        'phase': self.np_random.random() * math.pi * 2,
                        'max_radius': self.np_random.uniform(0.6, 0.9)
                    })
                    occupied_tiles.add(tuple(pos))
                    break

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        reward = -0.1  # Cost of living
        terminated = False

        self._update_jump_state(movement)
        self._update_shadows_and_items()
        self._update_particles()

        # If a jump just completed, check for events
        if self.jump_state['is_jumping'] and self.jump_state['progress'] >= self.JUMP_DURATION:
            self.jump_state['is_jumping'] = False
            self.player_pos = list(self.jump_state['target_pos'])

            # Check for item collection
            for item in self.items[:]:
                if self.player_pos == item['pos']:
                    self.score += 5
                    reward += 5
                    self.items.remove(item)
                    # sfx: item_pickup.wav
                    self._create_particles(self.player_pos, 20, self.COLOR_PARTICLE_ITEM)
                    break
            
            # Check for exit
            if self.player_pos == self.exit_pos:
                self.score += 100
                reward += 100
                terminated = True
                self.game_over_message = "ESCAPED!"
                # sfx: victory.wav
            
            # Check for shadow collision
            if not terminated:
                for shadow in self.shadows:
                    if shadow['pos'] == self.player_pos:
                        radius = shadow['max_radius'] * (0.6 + 0.4 * math.sin(shadow['phase']))
                        if radius > 0.4: # Player is a point, shadow needs to be big enough
                            self.score -= 50
                            reward -= 50
                            terminated = True
                            self.game_over_message = "CONSUMED BY SHADOW"
                            # sfx: player_death.wav
                            self._create_particles(self.player_pos, 30, self.COLOR_PARTICLE_DEATH, 5)
                            break
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over_message = "TIME RAN OUT"

        if self.steps > 0 and self.steps % self.SHADOW_SPEED_INCREASE_INTERVAL == 0:
            self.shadow_speed += self.SHADOW_SPEED_INCREASE_AMOUNT

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_jump_state(self, movement):
        if self.jump_state['is_jumping']:
            self.jump_state['progress'] += 1
        elif movement != 0: # Start a new jump
            target_pos = list(self.player_pos)
            if movement == 1: target_pos[1] -= 1 # Up (North-East)
            elif movement == 2: target_pos[1] += 1 # Down (South-West)
            elif movement == 3: target_pos[0] -= 1 # Left (North-West)
            elif movement == 4: target_pos[0] += 1 # Right (South-East)

            if 0 <= target_pos[0] < self.GRID_WIDTH and 0 <= target_pos[1] < self.GRID_HEIGHT:
                self.jump_state = {
                    'is_jumping': True,
                    'progress': 0,
                    'start_pos': self.player_pos,
                    'target_pos': target_pos
                }
                # sfx: player_jump.wav

    def _update_shadows_and_items(self):
        for shadow in self.shadows:
            shadow['phase'] += self.shadow_speed
        for item in self.items:
            item['pulse'] += 0.1

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _to_iso(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * self.TILE_WIDTH / 2
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * self.TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_game_objects()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                sx, sy = self._to_iso(x, y)
                points = [
                    (sx, sy),
                    (sx + self.TILE_WIDTH / 2, sy + self.TILE_HEIGHT / 2),
                    (sx, sy + self.TILE_HEIGHT),
                    (sx - self.TILE_WIDTH / 2, sy + self.TILE_HEIGHT / 2)
                ]
                pygame.draw.polygon(self.screen, self.COLOR_GRID, points, 1)

    def _render_game_objects(self):
        drawables = []
        # Add shadows
        for shadow in self.shadows:
            drawables.append({'type': 'shadow', 'data': shadow, 'pos': shadow['pos']})
        # Add items
        for item in self.items:
            drawables.append({'type': 'item', 'data': item, 'pos': item['pos']})
        # Add exit
        drawables.append({'type': 'exit', 'data': None, 'pos': self.exit_pos})
        # Add player shadow
        drawables.append({'type': 'player_shadow', 'data': None, 'pos': self.player_pos})
        # Add player
        drawables.append({'type': 'player', 'data': None, 'pos': self.player_pos})
        
        # Sort by grid y then x for proper occlusion
        drawables.sort(key=lambda d: (d['pos'][0] + d['pos'][1], d['pos'][1] - d['pos'][0]))

        for d in drawables:
            if d['type'] == 'shadow': self._render_a_shadow(d['data'])
            elif d['type'] == 'item': self._render_an_item(d['data'])
            elif d['type'] == 'exit': self._render_the_exit()
            elif d['type'] == 'player_shadow': self._render_the_player_shadow()
            elif d['type'] == 'player': self._render_the_player()

    def _render_a_shadow(self, shadow):
        sx, sy = self._to_iso(shadow['pos'][0], shadow['pos'][1])
        radius_factor = 0.6 + 0.4 * math.sin(shadow['phase'])
        radius_x = int(self.TILE_WIDTH / 2 * shadow['max_radius'] * radius_factor)
        radius_y = int(self.TILE_HEIGHT / 2 * shadow['max_radius'] * radius_factor)
        if radius_x > 0 and radius_y > 0:
            rect = pygame.Rect(sx - radius_x, sy + self.TILE_HEIGHT/2 - radius_y, radius_x*2, radius_y*2)
            pygame.gfxdraw.filled_ellipse(self.screen, rect.centerx, rect.centery, radius_x, radius_y, self.COLOR_SHADOW)
    
    def _render_an_item(self, item):
        sx, sy = self._to_iso(item['pos'][0], item['pos'][1])
        pulse_size = 2 + 2 * math.sin(item['pulse'])
        pygame.draw.circle(self.screen, self.COLOR_ITEM, (sx, sy), int(8 + pulse_size))
        # Glow effect
        glow_surf = pygame.Surface((32, 32), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_ITEM, 50), (16, 16), int(12 + pulse_size))
        self.screen.blit(glow_surf, (sx - 16, sy - 16), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_the_exit(self):
        sx, sy = self._to_iso(self.exit_pos[0], self.exit_pos[1])
        rect = pygame.Rect(sx - self.TILE_WIDTH/4, sy - self.TILE_HEIGHT/2, self.TILE_WIDTH/2, self.TILE_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, rect)
        pygame.draw.rect(self.screen, tuple(c*0.7 for c in self.COLOR_EXIT), rect, 2)

    def _render_the_player_shadow(self):
        px, py = self.player_pos
        if self.jump_state['is_jumping']:
            p = self.jump_state['progress'] / self.JUMP_DURATION
            start = np.array(self.jump_state['start_pos'])
            end = np.array(self.jump_state['target_pos'])
            interp_pos = start + (end - start) * p
            px, py = interp_pos[0], interp_pos[1]
        
        sx, sy = self._to_iso(px, py)
        shadow_surf = pygame.Surface((self.TILE_WIDTH, self.TILE_HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_ellipse(shadow_surf, self.TILE_WIDTH//2, self.TILE_HEIGHT//2, 8, 4, self.COLOR_PLAYER_SHADOW)
        self.screen.blit(shadow_surf, (sx - self.TILE_WIDTH//2, sy))

    def _render_the_player(self):
        px, py = self.player_pos
        z_offset = 0
        if self.jump_state['is_jumping']:
            p = self.jump_state['progress'] / self.JUMP_DURATION
            start = np.array(self.jump_state['start_pos'])
            end = np.array(self.jump_state['target_pos'])
            interp_pos = start + (end - start) * p
            px, py = interp_pos[0], interp_pos[1]
            # Parabolic jump arc
            z_offset = self.JUMP_HEIGHT * (4 * p * (1 - p))

        sx, sy = self._to_iso(px, py)
        sy -= int(z_offset)
        
        # Player glow
        glow_surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER, 30), (20, 20), 12)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER, 15), (20, 20), 18)
        self.screen.blit(glow_surf, (sx - 20, sy - 20), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Player core
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (sx, sy), 6)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (size, size), size)
                self.screen.blit(surf, (p['pos'][0]-size, p['pos'][1]-size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Steps
        steps_text = self.font_large.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over_message:
            msg_surf = self.font_large.render(self.game_over_message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0,150), msg_rect.inflate(20, 20))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "items_left": len(self.items)
        }

    def _create_particles(self, grid_pos, count, color, speed_mult=1):
        sx, sy = self._to_iso(grid_pos[0], grid_pos[1])
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 * speed_mult + 1
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': [sx, sy],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def close(self):
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=42)
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=random.randint(0, 10000))
    
    pygame.display.set_caption("Shadow Jumper")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    action = [0, 0, 0] # no-op, no space, no shift
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        current_action = [movement, space_held, shift_held]

        # Since auto_advance is False, we only step when an action is taken
        # For human play, this means on every frame where a key is pressed OR if the player is jumping
        if movement != 0 or env.jump_state['is_jumping']:
            obs, reward, terminated, truncated, info = env.step(current_action)
        
        # Draw the observation to the screen
        draw_obs = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(screen, draw_obs)
        pygame.display.flip()

        env.clock.tick(env.metadata["render_fps"])

    print(f"Game Over! Final Score: {info['score']}")
    pygame.time.wait(2000) # Wait 2 seconds before closing
    env.close()