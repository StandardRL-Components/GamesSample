import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:35:01.764524
# Source Brief: brief_00541.md
# Brief Index: 541
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a neon maze game.
    The agent controls a character that can move, flip gravity, and rewind time
    for moving platforms. The goal is to reach the exit while avoiding dripping horrors.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Navigate a neon maze to reach the exit, avoiding dripping horrors. "
        "Manipulate time and gravity to overcome moving obstacles."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press Shift to flip gravity and Space to rewind platform positions."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PLAYER_SIZE = 16
        self.PLAYER_SPEED = 5
        self.EXIT_SIZE = 30
        self.MAX_STEPS = 2000
        self.PLATFORM_HISTORY_LENGTH = 60 # Store 2 seconds of history at 30fps

        # --- Colors ---
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255)
        self.COLOR_EXIT = (255, 255, 255)
        self.COLOR_EXIT_GLOW = (220, 220, 100)
        self.COLOR_PLATFORMS = [
            (0, 255, 255), (255, 0, 255), (0, 255, 0), 
            (255, 128, 0), (128, 0, 255)
        ]
        self.COLOR_HORROR = (255, 50, 50)
        self.COLOR_HORROR_GLOW = (200, 0, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_GRAVITY_UP = (100, 100, 255)
        self.COLOR_GRAVITY_DOWN = (255, 100, 100)
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)
        
        # --- State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.player_rect = pygame.Rect(0, 0, 0, 0)
        self.exit_pos = pygame.Vector2(0, 0)
        self.exit_rect = pygame.Rect(0, 0, 0, 0)
        self.platforms = []
        self.horrors = []
        self.particles = []
        self.gravity_down = True
        self.horror_speed = 0.0
        self.horror_spawn_timer = 0
        self.platform_history = collections.deque(maxlen=self.PLATFORM_HISTORY_LENGTH)
        self.prev_space_held = False
        self.prev_shift_held = False
        self.rewind_effect_timer = 0
        self.gravity_flip_effect_timer = 0
        
        # self.reset() is called by the wrapper, no need to call it here
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player and Exit
        self.player_pos = pygame.Vector2(self.PLAYER_SIZE * 2, self.HEIGHT / 2)
        self.exit_pos = pygame.Vector2(self.WIDTH - self.EXIT_SIZE * 2, self.HEIGHT / 2)
        self.exit_rect = pygame.Rect(self.exit_pos.x, self.exit_pos.y, self.EXIT_SIZE, self.EXIT_SIZE)
        
        # Game mechanics state
        self.gravity_down = True
        self.horror_speed = 1.0
        self.horror_spawn_timer = 60
        
        # Clear dynamic lists
        self.horrors.clear()
        self.particles.clear()
        self.platform_history.clear()

        # Input state
        self.prev_space_held = False
        self.prev_shift_held = False

        # Visual effect timers
        self.rewind_effect_timer = 0
        self.gravity_flip_effect_timer = 0
        
        self._generate_platforms()

        return self._get_observation(), self._get_info()

    def _generate_platforms(self):
        self.platforms.clear()
        num_platforms = self.np_random.integers(6, 9)
        for i in range(num_platforms):
            # Ensure platforms are not too close to start/end points
            px = self.np_random.uniform(self.WIDTH * 0.15, self.WIDTH * 0.85)
            py = self.np_random.uniform(0, self.HEIGHT)
            
            is_vertical = self.np_random.choice([True, False])
            if is_vertical:
                w, h = 15, self.np_random.uniform(80, 150)
                move_range = self.np_random.uniform(50, self.HEIGHT / 2 - h / 2)
                move_axis = 'y'
            else:
                w, h = self.np_random.uniform(80, 150), 15
                move_range = self.np_random.uniform(50, self.WIDTH / 2 - w / 2)
                move_axis = 'x'

            platform = {
                'rect': pygame.Rect(px - w/2, py - h/2, w, h),
                'color': random.choice(self.COLOR_PLATFORMS),
                'center': pygame.Vector2(px, py),
                'range': move_range,
                'speed': self.np_random.uniform(0.01, 0.03),
                'offset': self.np_random.uniform(0, 2 * math.pi),
                'axis': move_axis
            }
            self.platforms.append(platform)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        dist_before = self.player_pos.distance_to(self.exit_pos)
        
        self._handle_input(movement, space_held, shift_held)
        self._update_player_position()
        self._update_platforms()
        self._update_horrors()
        self._update_particles()
        self._update_effects()

        self.steps += 1
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.horror_speed += 0.05
        
        terminated, term_reward = self._check_collisions()
        
        # Calculate reward
        dist_after = self.player_pos.distance_to(self.exit_pos)
        reward = (dist_before - dist_after) * 0.01 # Scaled down to be small
        reward += term_reward
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated and not terminated:
            terminated = True # Environment terminates on timeout
        
        self.score += reward
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # --- Player Movement ---
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1  # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1  # Right
        
        if not self.gravity_down:
            move_vec.y *= -1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
        self.player_pos += move_vec * self.PLAYER_SPEED

        # --- Gravity Flip ---
        if shift_held and not self.prev_shift_held:
            self.gravity_down = not self.gravity_down
            self.gravity_flip_effect_timer = 15
            self._create_particles(self.player_pos, 20, self.COLOR_GRAVITY_UP if not self.gravity_down else self.COLOR_GRAVITY_DOWN)

        # --- Time Rewind ---
        if space_held and not self.prev_space_held:
            if len(self.platform_history) > 1:
                self.platform_history.pop() # Remove current state
                last_states = self.platform_history.pop()
                for i, platform in enumerate(self.platforms):
                    platform['rect'].topleft = last_states[i]
                self.rewind_effect_timer = 10
                self._create_particles(self.player_pos, 10, self.COLOR_PLAYER_GLOW)
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_player_position(self):
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE / 2, self.WIDTH - self.PLAYER_SIZE / 2)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE / 2, self.HEIGHT - self.PLAYER_SIZE / 2)
        self.player_rect.center = self.player_pos

    def _update_platforms(self):
        # Store current state for rewind
        current_platform_positions = [p['rect'].topleft for p in self.platforms]
        self.platform_history.append(current_platform_positions)

        # Move platforms
        for p in self.platforms:
            oscillation = math.sin(self.steps * p['speed'] + p['offset']) * p['range']
            if p['axis'] == 'x':
                p['rect'].centerx = p['center'].x + oscillation
            else:
                p['rect'].centery = p['center'].y + oscillation

    def _update_horrors(self):
        self.horror_spawn_timer -= 1
        if self.horror_spawn_timer <= 0:
            spawn_x = self.np_random.uniform(0, self.WIDTH)
            spawn_y = -10 if self.gravity_down else self.HEIGHT + 10
            size = self.np_random.uniform(8, 14)
            self.horrors.append({
                'pos': pygame.Vector2(spawn_x, spawn_y),
                'size': size,
                'trail': collections.deque(maxlen=10)
            })
            self.horror_spawn_timer = max(15, 60 - self.steps // 50) # Spawn faster over time

        drip_direction = 1 if self.gravity_down else -1
        for horror in self.horrors[:]:
            horror['trail'].append(horror['pos'].copy())
            horror['pos'].y += self.horror_speed * drip_direction
            if not (0 < horror['pos'].y < self.HEIGHT):
                self.horrors.remove(horror)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _update_effects(self):
        if self.rewind_effect_timer > 0:
            self.rewind_effect_timer -= 1
        if self.gravity_flip_effect_timer > 0:
            self.gravity_flip_effect_timer -= 1

    def _check_collisions(self):
        # Player is centered, rect needs to be updated
        self.player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.player_rect.center = self.player_pos

        # Player vs. Horrors
        for horror in self.horrors:
            horror_rect = pygame.Rect(horror['pos'].x - horror['size']/2, horror['pos'].y - horror['size']/2, horror['size'], horror['size'])
            if self.player_rect.colliderect(horror_rect):
                self._create_particles(self.player_pos, 50, self.COLOR_HORROR)
                return True, -100.0 # Terminated, reward
        
        # Player vs. Exit
        if self.player_rect.colliderect(self.exit_rect):
            self._create_particles(self.exit_rect.center, 100, self.COLOR_EXIT_GLOW)
            return True, 105.0 # Terminated, reward (+100 win, +5 reach)

        # Player vs. Platforms
        for p in self.platforms:
            if self.player_rect.colliderect(p['rect']):
                # Simple push-out collision response
                overlap = self.player_rect.clip(p['rect'])
                if overlap.width < overlap.height:
                    if self.player_rect.centerx < p['rect'].centerx:
                        self.player_pos.x -= overlap.width
                    else:
                        self.player_pos.x += overlap.width
                else:
                    if self.player_rect.centery < p['rect'].centery:
                        self.player_pos.y -= overlap.height
                    else:
                        self.player_pos.y += overlap.height
                self.player_rect.center = self.player_pos
        
        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over_screen()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Effects ---
        if self.rewind_effect_timer > 0:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = int(70 * (self.rewind_effect_timer / 10))
            s.fill((200, 200, 255, alpha))
            self.screen.blit(s, (0, 0))

        if self.gravity_flip_effect_timer > 0:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = int(90 * (self.gravity_flip_effect_timer / 15))
            color = self.COLOR_GRAVITY_UP if not self.gravity_down else self.COLOR_GRAVITY_DOWN
            s.fill((*color, alpha))
            self.screen.blit(s, (0, 0))

        # --- Particles ---
        for p in self.particles:
            alpha = max(0, 255 * (p['lifespan'] / p['max_life']))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])), special_flags=pygame.BLEND_RGBA_ADD)

        # --- Exit ---
        self._draw_glow_rect(self.exit_rect, self.COLOR_EXIT, self.COLOR_EXIT_GLOW, 15)
        
        # --- Platforms ---
        for p in self.platforms:
            pygame.draw.rect(self.screen, p['color'], p['rect'], border_radius=3)
            # Add a subtle inner glow
            inner_rect = p['rect'].inflate(-6, -6)
            s = pygame.Surface(p['rect'].size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_PLAYER, 30), s.get_rect(), border_radius=3)
            self.screen.blit(s, p['rect'].topleft, special_flags=pygame.BLEND_RGBA_ADD)

        # --- Horrors ---
        for horror in self.horrors:
            # Trail
            for i, pos in enumerate(horror['trail']):
                alpha = int(150 * (i / len(horror['trail'])))
                radius = int(horror['size']/2 * (i / len(horror['trail'])))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, (*self.COLOR_HORROR_GLOW, alpha))
            # Main body
            self._draw_glow_circle(horror['pos'], horror['size']/2, self.COLOR_HORROR, self.COLOR_HORROR_GLOW, 10)

        # --- Player ---
        player_render_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_render_rect.center = self.player_pos
        self._draw_glow_rect(player_render_rect, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, 20)

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Steps
        steps_text = self.font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        # Gravity Indicator
        indicator_size = 20
        indicator_rect = pygame.Rect(self.WIDTH/2 - indicator_size/2, 10, indicator_size, indicator_size)
        color = self.COLOR_GRAVITY_DOWN if self.gravity_down else self.COLOR_GRAVITY_UP
        if self.gravity_down:
            points = [(indicator_rect.left, indicator_rect.top), (indicator_rect.right, indicator_rect.top), (indicator_rect.centerx, indicator_rect.bottom)]
        else:
            points = [(indicator_rect.left, indicator_rect.bottom), (indicator_rect.right, indicator_rect.bottom), (indicator_rect.centerx, indicator_rect.top)]
        pygame.draw.polygon(self.screen, color, points)

    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text = "EPISODE END"
        text_surf = self.font_big.render(text, True, self.COLOR_PLAYER)
        text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text_surf, text_rect)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": (self.player_pos.x, self.player_pos.y),
            "exit_pos": (self.exit_pos.x, self.exit_pos.y),
            "horror_count": len(self.horrors),
        }

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'lifespan': lifespan,
                'max_life': lifespan,
                'color': color,
                'size': self.np_random.uniform(1, 4)
            })

    def _draw_glow_rect(self, rect, color, glow_color, glow_size):
        for i in range(glow_size, 0, -2):
            alpha = int(100 * (1 - i / glow_size))
            s = pygame.Surface((rect.width + i, rect.height + i), pygame.SRCALPHA)
            pygame.draw.rect(s, (*glow_color, alpha), s.get_rect(), border_radius=int(i/2))
            self.screen.blit(s, (rect.left - i/2, rect.top - i/2), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.draw.rect(self.screen, color, rect, border_radius=3)
        
    def _draw_glow_circle(self, pos, radius, color, glow_color, glow_size):
        int_pos = (int(pos.x), int(pos.y))
        for i in range(glow_size, 0, -2):
            alpha = int(120 * (1 - i / glow_size))
            pygame.gfxdraw.filled_circle(self.screen, int_pos[0], int_pos[1], int(radius + i/2), (*glow_color, alpha))
        pygame.gfxdraw.filled_circle(self.screen, int_pos[0], int_pos[1], int(radius), color)
        pygame.gfxdraw.aacircle(self.screen, int_pos[0], int_pos[1], int(radius), color)

    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run with the "dummy" video driver, so we unset it.
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Neon Maze")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        movement = 0 # None
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, term, trunc, info = env.step(action)
        terminated = term
        truncated = trunc
        
        # Convert observation back to a Surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.metadata['render_fps'])

    env.close()