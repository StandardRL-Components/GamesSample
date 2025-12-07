import gymnasium as gym
import os
import pygame
import numpy as np
import math
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a surreal dream-diving game.
    The player controls a diver, flipping gravity to navigate a descending
    dreamscape, collecting shards and avoiding obstacles.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Navigate a surreal dreamscape by flipping gravity. Collect glowing shards while avoiding crystalline obstacles in a rapid descent."
    user_guide = "Use ←→ arrow keys to move horizontally. Press space to flip gravity and navigate vertically."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30
    MAX_DEPTH = 10000
    VICTORY_DEPTH = 10000
    MAX_STEPS = 5000

    # Colors
    COLOR_BG_TOP = (10, 20, 50)
    COLOR_BG_BOTTOM = (0, 5, 15)
    COLOR_PLAYER = (255, 255, 100)
    COLOR_PLAYER_GLOW = (255, 255, 100, 30)
    COLOR_OBSTACLE = (100, 80, 180)
    COLOR_OBSTACLE_GLOW = (100, 80, 180, 20)
    COLOR_SHARD_YELLOW = (255, 250, 150)
    COLOR_SHARD_PINK = (255, 150, 250)
    COLOR_SHARD_CYAN = (150, 255, 250)
    COLOR_TEXT = (230, 230, 240)
    COLOR_TEXT_SHADOW = (20, 20, 30)
    COLOR_BAR_BG = (50, 50, 80)
    COLOR_BAR_FILL = (150, 200, 255)

    # Player Physics
    PLAYER_H_ACCEL = 1.2
    PLAYER_H_FRICTION = 0.85
    PLAYER_MAX_H_SPEED = 8.0
    PLAYER_GRAVITY = 0.6
    PLAYER_SIZE = (12, 20)

    # Game Dynamics
    WORLD_SCROLL_SPEED = 2.0
    GRAVITY_FLIP_COOLDOWN = 15 # frames
    OBSTACLE_BASE_SPEED = 1.0
    OBSTACLE_BASE_SPAWN_CHANCE = 0.02
    SHARD_SPAWN_CHANCE = 0.03

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 14)
        
        # Internal state variables
        self.player_pos = None
        self.player_vel = None
        self.gravity_direction = None
        self.gravity_flip_cooldown_timer = None
        self.depth = None
        self.score = None
        self.shards_collected = None
        self.steps = None
        self.game_over = None
        self.obstacles = None
        self.shards = None
        self.particles = None
        self.bg_gradient = None
        self.last_reward_depth = None
        self.last_space_action = 0

        self._create_bg_gradient()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT * 0.2])
        self.player_vel = np.array([0.0, 0.0])
        self.gravity_direction = 1  # 1 for down, -1 for up
        self.gravity_flip_cooldown_timer = 0
        
        self.depth = 0.0
        self.score = 0
        self.shards_collected = 0
        self.steps = 0
        self.game_over = False
        self.last_reward_depth = 0
        self.last_space_action = 0

        self.obstacles = []
        self.shards = []
        self.particles = []
        
        # Pre-populate the world
        for _ in range(15):
            self._spawn_shard(self.np_random.uniform(0, self.SCREEN_HEIGHT))
        for _ in range(10):
            self._spawn_obstacle(self.np_random.uniform(0, self.SCREEN_HEIGHT))

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        self._handle_input(action)
        self._update_game_state()
        
        # --- Calculate Rewards ---
        # Reward for depth progression
        depth_gained = self.depth - self.last_reward_depth
        if depth_gained > 0:
            reward += depth_gained * 0.1
            self.last_reward_depth = self.depth

        # Collision checks and event rewards
        reward += self._check_collisions()
        
        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if self.game_over: # Set by collision check
            reward -= 100 # Collision penalty
            terminated = True
        elif self.depth >= self.VICTORY_DEPTH:
            reward += 100 # Victory bonus
            terminated = True
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )
        
    def _handle_input(self, action):
        movement, space_action, _ = action
        
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel[0] -= self.PLAYER_H_ACCEL
        elif movement == 4:  # Right
            self.player_vel[0] += self.PLAYER_H_ACCEL
            
        # Gravity Flip on button press (transition from 0 to 1)
        if space_action == 1 and self.last_space_action == 0 and self.gravity_flip_cooldown_timer <= 0:
            self.gravity_direction *= -1
            self.gravity_flip_cooldown_timer = self.GRAVITY_FLIP_COOLDOWN
            # Sound placeholder: "sfx_gravity_flip.wav"
            self._create_particles(self.player_pos, 20, (200, 200, 255), 5, 20)

        self.last_space_action = space_action

    def _update_game_state(self):
        # Update timers
        if self.gravity_flip_cooldown_timer > 0:
            self.gravity_flip_cooldown_timer -= 1

        # Update player velocity
        self.player_vel[0] *= self.PLAYER_H_FRICTION
        self.player_vel[0] = np.clip(self.player_vel[0], -self.PLAYER_MAX_H_SPEED, self.PLAYER_MAX_H_SPEED)
        self.player_vel[1] += self.PLAYER_GRAVITY * self.gravity_direction
        
        # Update player position
        self.player_pos += self.player_vel

        # World scroll and depth update
        scroll_speed = self.WORLD_SCROLL_SPEED + max(0, self.player_vel[1] * self.gravity_direction * 0.1)
        self.depth += scroll_speed
        
        # Move all entities with the world scroll
        self.player_pos[1] -= scroll_speed
        for o in self.obstacles: o['pos'][1] -= scroll_speed
        for s in self.shards: s['pos'][1] -= scroll_speed
        for p in self.particles: p['pos'][1] -= scroll_speed

        # Player boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.SCREEN_WIDTH)
        
        # Bounce off top/bottom to prevent termination in stable no-op test
        player_half_h = self.PLAYER_SIZE[1] / 2
        if self.player_pos[1] < player_half_h:
            self.player_pos[1] = player_half_h
            if self.player_vel[1] < 0:
                self.player_vel[1] *= -0.5
        elif self.player_pos[1] > self.SCREEN_HEIGHT - player_half_h:
            self.player_pos[1] = self.SCREEN_HEIGHT - player_half_h
            if self.player_vel[1] > 0:
                self.player_vel[1] *= -0.5

        # Update difficulty based on depth
        difficulty_tier = int(self.depth / 500)
        current_obstacle_speed = self.OBSTACLE_BASE_SPEED + difficulty_tier * 0.05
        current_obstacle_spawn_chance = self.OBSTACLE_BASE_SPAWN_CHANCE + difficulty_tier * 0.001

        # Update obstacles
        for o in self.obstacles:
            o['pos'][0] += o['vel'][0] * current_obstacle_speed
            o['pos'][1] += o['vel'][1] * current_obstacle_speed
            o['angle'] += o['rot_speed']
            if o['pos'][0] < -50 or o['pos'][0] > self.SCREEN_WIDTH + 50:
                o['vel'][0] *= -1 # Bounce horizontally
        
        # Update shards (pulsing effect)
        for s in self.shards:
            s['pulse_timer'] += 0.1
            s['size'] = s['base_size'] + math.sin(s['pulse_timer']) * 2
            
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            
        # Prune off-screen entities and spawn new ones
        self.obstacles = [o for o in self.obstacles if o['pos'][1] > -50]
        self.shards = [s for s in self.shards if s['pos'][1] > -50]
        
        if self.np_random.random() < current_obstacle_spawn_chance:
            self._spawn_obstacle()
        if self.np_random.random() < self.SHARD_SPAWN_CHANCE:
            self._spawn_shard()

    def _check_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE[0]/2,
                                  self.player_pos[1] - self.PLAYER_SIZE[1]/2,
                                  *self.PLAYER_SIZE)

        # Obstacles
        for o in self.obstacles:
            obstacle_rect = pygame.Rect(o['pos'][0] - o['size']/2, o['pos'][1] - o['size']/2, o['size'], o['size'])
            if player_rect.colliderect(obstacle_rect):
                self.game_over = True
                # Sound placeholder: "sfx_collision_death.wav"
                self._create_particles(self.player_pos, 50, self.COLOR_PLAYER, 8, 40)
                break
        
        # Shards
        collected_shards = []
        for i, s in enumerate(self.shards):
            shard_rect = pygame.Rect(s['pos'][0] - s['size']/2, s['pos'][1] - s['size']/2, s['size'], s['size'])
            if player_rect.colliderect(shard_rect):
                collected_shards.append(i)
                self.score += 1
                self.shards_collected += 1
                reward += 1
                # Sound placeholder: "sfx_collect_shard.wav"
                self._create_particles(s['pos'], 15, s['color'], 3, 30)

        # Remove collected shards (in reverse to avoid index errors)
        for i in sorted(collected_shards, reverse=True):
            del self.shards[i]
            
        return reward

    def _spawn_obstacle(self, y_pos=None):
        if y_pos is None:
            y_pos = self.SCREEN_HEIGHT + 40
        
        side = self.np_random.choice([-1, 1])
        x_pos = self.SCREEN_WIDTH / 2 + side * (self.SCREEN_WIDTH / 2 + 30)
        
        self.obstacles.append({
            'pos': np.array([x_pos, y_pos], dtype=float),
            'vel': np.array([-side, self.np_random.uniform(-0.2, 0.2)], dtype=float),
            'size': self.np_random.uniform(30, 80),
            'angle': self.np_random.uniform(0, 360),
            'rot_speed': self.np_random.uniform(-0.5, 0.5)
        })

    def _spawn_shard(self, y_pos=None):
        if y_pos is None:
            y_pos = self.SCREEN_HEIGHT + 20
        
        base_size = self.np_random.uniform(8, 12)
        self.shards.append({
            'pos': np.array([self.np_random.uniform(20, self.SCREEN_WIDTH - 20), y_pos], dtype=float),
            'color': self.np_random.choice([self.COLOR_SHARD_YELLOW, self.COLOR_SHARD_PINK, self.COLOR_SHARD_CYAN], p=[0.34, 0.33, 0.33]),
            'base_size': base_size,
            'size': base_size,
            'pulse_timer': self.np_random.uniform(0, math.pi * 2)
        })

    def _create_particles(self, pos, count, color, max_speed, max_life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': self.np_random.integers(max_life // 2, max_life),
                'max_life': max_life,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })
            
    def _create_bg_gradient(self):
        self.bg_gradient = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.bg_gradient, color, (0, y), (self.SCREEN_WIDTH, y))

    def _get_observation(self):
        # Clear screen with background
        self.screen.blit(self.bg_gradient, (0, 0))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {
            "score": self.score,
            "shards_collected": self.shards_collected,
            "steps": self.steps,
            "depth": self.depth,
            "player_pos": self.player_pos.tolist(),
            "player_vel": self.player_vel.tolist(),
        }
        
    def _render_game(self):
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p['color'], alpha), (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Shards
        for s in self.shards:
            self._draw_glow_circle(self.screen, s['pos'], s['size'], s['color'])

        # Obstacles
        for o in self.obstacles:
            self._draw_glowing_polygon(self.screen, o['pos'], o['size'], 6, o['angle'], self.COLOR_OBSTACLE)

        # Player
        if not self.game_over:
            player_rect = pygame.Rect(0, 0, *self.PLAYER_SIZE)
            player_rect.center = tuple(map(int, self.player_pos))
            
            # Glow effect
            glow_surf = pygame.Surface((self.PLAYER_SIZE[0] * 3, self.PLAYER_SIZE[1] * 3), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect(), border_radius=8)
            self.screen.blit(glow_surf, (player_rect.centerx - glow_surf.get_width()/2, player_rect.centery - glow_surf.get_height()/2), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Main body
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
            
            # Gravity indicator
            indicator_y = player_rect.top - 5 if self.gravity_direction == -1 else player_rect.bottom + 5
            indicator_points = [
                (player_rect.centerx, indicator_y),
                (player_rect.centerx - 4, indicator_y - 4 * self.gravity_direction),
                (player_rect.centerx + 4, indicator_y - 4 * self.gravity_direction)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, indicator_points)

    def _render_ui(self):
        # Depth
        self._draw_text(f"DEPTH: {int(self.depth)} m", (10, 5), self.font_main)
        # Shards
        self._draw_text(f"SHARDS: {self.shards_collected}", (self.SCREEN_WIDTH - 10, 5), self.font_main, align="right")
        # Depth/Oxygen bar
        bar_width = 200
        bar_height = 15
        bar_x = self.SCREEN_WIDTH / 2 - bar_width / 2
        bar_y = 10
        fill_ratio = min(1.0, self.depth / self.MAX_DEPTH)
        
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_BAR_FILL, (bar_x, bar_y, bar_width * fill_ratio, bar_height), border_radius=3)

    def _draw_text(self, text, pos, font, align="left"):
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surf = font.render(text, True, self.COLOR_TEXT)
        
        if align == "right":
            pos = (pos[0] - text_surf.get_width(), pos[1])
        elif align == "center":
            pos = (pos[0] - text_surf.get_width() / 2, pos[1])

        self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
        self.screen.blit(text_surf, pos)

    def _draw_glow_circle(self, surface, pos, radius, color):
        x, y = int(pos[0]), int(pos[1])
        # Outer glow
        glow_radius = int(radius * 2.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*color, 20))
        surface.blit(glow_surf, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        # Inner glow
        glow_radius = int(radius * 1.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*color, 40))
        surface.blit(glow_surf, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        # Core circle
        pygame.gfxdraw.filled_circle(surface, x, y, int(radius), color)
        pygame.gfxdraw.aacircle(surface, x, y, int(radius), color)

    def _draw_glowing_polygon(self, surface, pos, size, num_sides, angle, color):
        x, y = int(pos[0]), int(pos[1])
        points = []
        for i in range(num_sides):
            rad = math.radians(angle + (360 / num_sides) * i)
            px = x + math.cos(rad) * size / 2
            py = y + math.sin(rad) * size / 2
            points.append((int(px), int(py)))
        
        # Glow
        glow_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        glow_points = [(p[0] - x + size, p[1] - y + size) for p in points]
        pygame.gfxdraw.filled_polygon(glow_surf, glow_points, self.COLOR_OBSTACLE_GLOW)
        surface.blit(glow_surf, (x - size, y - size), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Core shape
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def close(self):
        pygame.quit()


# --- Example Usage ---
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # --- Manual Play Setup ---
    # This part will raise an error in a headless environment.
    # It is intended for local testing with a display.
    try:
        pygame.display.init()
        pygame.display.set_caption("Dream Dive")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    except pygame.error:
        print("Pygame display could not be initialized. Running headlessly.")
        screen = None

    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    terminated = False
    
    while running:
        # --- Pygame Event Handling for Manual Control ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if terminated:
            if screen:
                # Draw game over screen
                font_game_over = pygame.font.SysFont("Consolas", 40, bold=True)
                # Use a temporary surface to draw game over text on the last observation
                game_over_obs = obs.copy()
                temp_surf = pygame.surfarray.make_surface(np.transpose(game_over_obs, (1, 0, 2)))
                env.screen.blit(temp_surf, (0,0))
                env._draw_text("GAME OVER", (env.SCREEN_WIDTH/2, env.SCREEN_HEIGHT/2 - 30), font_game_over, align="center")
                env._draw_text(f"Final Score: {int(env.score)}", (env.SCREEN_WIDTH/2, env.SCREEN_HEIGHT/2 + 20), env.font_main, align="center")
                env._draw_text("Press 'R' to restart", (env.SCREEN_WIDTH/2, env.SCREEN_HEIGHT/2 + 50), env.font_main, align="center")
                screen.blit(env.screen, (0, 0))
                pygame.display.flip()
            continue

        # --- Action Mapping for Manual Control ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        if screen:
            frame_to_show = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame_to_show)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        clock.tick(GameEnv.TARGET_FPS)

    env.close()