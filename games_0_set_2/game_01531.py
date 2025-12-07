import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys for cardinal jumps. Space for up-right jump, Shift for down-right jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Leap between procedurally generated isometric platforms, dodging enemies, to reach the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_WIDTH, self.WORLD_HEIGHT = 30, 60
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_PLAYER_SIDE = (200, 150, 0)
        self.COLOR_SHADOW = (0, 0, 0, 50)
        self.COLOR_ENEMY = (200, 50, 220)
        self.COLOR_ENEMY_SIDE = (150, 40, 170)
        self.COLOR_GREEN_PLATFORM = (0, 200, 120)
        self.COLOR_GREEN_SIDE = (0, 150, 90)
        self.COLOR_RED_PLATFORM = (255, 80, 80)
        self.COLOR_RED_SIDE = (200, 60, 60)
        self.COLOR_TOP_PLATFORM = (200, 200, 255)
        self.COLOR_TOP_SIDE = (150, 150, 200)
        self.COLOR_UI_TEXT = (240, 240, 240)

        # Physics and game parameters
        self.GRAVITY = -0.3
        self.JUMP_Z_VEL = 6
        self.JUMP_XY_VEL = 1.3
        self.MAX_STEPS = 2000
        self.ISO_TILE_WIDTH_HALF = 16
        self.ISO_TILE_HEIGHT_HALF = 8

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_on_ground = None
        self.player_jump_cooldown = None
        self.platforms = None
        self.enemies = None
        self.particles = None
        self.camera_y = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.enemy_speed = None
        self.platform_spacing = None
        self.top_platform_y = None
        
        # self.reset() is called by the wrapper, but for standalone use it's good practice
        # to have a fully initialized object after __init__
        # self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.player_pos = np.array([self.WORLD_WIDTH / 2, 5, 10], dtype=float)
        self.player_vel = np.array([0.0, 0.0, 0.0], dtype=float)
        self.player_on_ground = False
        self.player_jump_cooldown = 0

        self.platforms = []
        self.enemies = []
        self.particles = []

        self.camera_y = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.enemy_speed = 0.5
        self.platform_spacing = 30
        self.top_platform_y = 500

        # Create starting and top platforms
        start_platform = self._create_platform(self.player_pos[0], self.player_pos[1], 0, 'green')
        start_platform['z'] = 0
        self.platforms.append(start_platform)
        self.player_pos[2] = start_platform['pos'][2] + start_platform['depth']
        self.player_on_ground = True

        top_platform = self._create_platform(self.WORLD_WIDTH / 2, self.top_platform_y, 0, 'top')
        top_platform['w'] = self.WORLD_WIDTH
        self.platforms.append(top_platform)

        # Procedurally generate initial platforms
        self._generate_world()
        
        return self._get_observation(), self._get_info()

    def _create_platform(self, x, y, z, p_type):
        w = self.np_random.integers(4, 8)
        h = self.np_random.integers(4, 8)
        depth = self.np_random.integers(5, 15)
        color, side_color = {
            'green': (self.COLOR_GREEN_PLATFORM, self.COLOR_GREEN_SIDE),
            'red': (self.COLOR_RED_PLATFORM, self.COLOR_RED_SIDE),
            'top': (self.COLOR_TOP_PLATFORM, self.COLOR_TOP_SIDE)
        }[p_type]
        
        return {
            'pos': np.array([x, y, z], dtype=float),
            'w': w, 'h': h, 'depth': depth,
            'type': p_type, 'color': color, 'side_color': side_color
        }

    def _generate_world(self):
        last_y = 0
        while last_y < self.top_platform_y - 50:
            y_offset = last_y + self.np_random.uniform(20, self.platform_spacing)
            x_offset = self.np_random.uniform(5, self.WORLD_WIDTH - 5)
            z_offset = self.np_random.uniform(0, 40)
            
            p_type = 'red' if self.np_random.random() < 0.2 else 'green'
            new_platform = self._create_platform(x_offset, y_offset, z_offset, p_type)
            self.platforms.append(new_platform)

            # Chance to spawn an enemy on a green platform
            if p_type == 'green' and self.np_random.random() < 0.3:
                self._spawn_enemy(len(self.platforms) - 1)
            
            last_y = y_offset

    def _spawn_enemy(self, platform_idx):
        platform = self.platforms[platform_idx]
        enemy_pos = np.copy(platform['pos'])
        enemy_pos[2] += platform['depth']
        self.enemies.append({
            'pos': enemy_pos,
            'platform_idx': platform_idx,
            'dir': self.np_random.choice([-1, 1]),
            'size': 2.0
        })

    def _iso_to_screen(self, x, y, z):
        screen_x = self.WIDTH / 2 + (x - y) * self.ISO_TILE_WIDTH_HALF
        screen_y = self.HEIGHT * 0.2 + (x + y) * self.ISO_TILE_HEIGHT_HALF - z - self.camera_y
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, x, y, z, w, h, depth, top_color, side_color):
        points = [
            self._iso_to_screen(x, y, z),
            self._iso_to_screen(x + w, y, z),
            self._iso_to_screen(x + w, y + h, z),
            self._iso_to_screen(x, y + h, z),
            self._iso_to_screen(x, y, z + depth),
            self._iso_to_screen(x + w, y, z + depth),
            self._iso_to_screen(x + w, y + h, z + depth),
            self._iso_to_screen(x, y + h, z + depth),
        ]
        
        # Draw sides first
        # Right side
        pygame.gfxdraw.filled_polygon(surface, [points[1], points[2], points[6], points[5]], side_color)
        pygame.gfxdraw.aapolygon(surface, [points[1], points[2], points[6], points[5]], side_color)
        # Left side
        pygame.gfxdraw.filled_polygon(surface, [points[2], points[3], points[7], points[6]], side_color)
        pygame.gfxdraw.aapolygon(surface, [points[2], points[3], points[7], points[6]], side_color)
        # Top
        pygame.gfxdraw.filled_polygon(surface, [points[4], points[5], points[6], points[7]], top_color)
        pygame.gfxdraw.aapolygon(surface, [points[4], points[5], points[6], points[7]], top_color)

    def _create_particles(self, pos, count, color, speed, life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = np.array([math.cos(angle), math.sin(angle), self.np_random.uniform(0.5, 2)]) * speed
            self.particles.append({'pos': np.copy(pos), 'vel': vel, 'life': life, 'color': color})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # Unpack factorized action and handle jump logic
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        old_y_pos = self.player_pos[1]

        if self.player_jump_cooldown > 0:
            self.player_jump_cooldown -= 1
        
        jump_dir = np.array([0.0, 0.0])
        jump_executed = False
        if self.player_on_ground and self.player_jump_cooldown == 0:
            # Action priority: space > shift > movement
            if space_held: # Up-Right
                jump_dir = np.array([1.0, 1.0])
                jump_executed = True
            elif shift_held: # Down-Right
                jump_dir = np.array([1.0, -1.0])
                jump_executed = True
            elif movement != 0:
                if movement == 1: jump_dir = np.array([-1.0, 1.0]) # Up-Left
                elif movement == 2: jump_dir = np.array([-1.0, -1.0]) # Down-Left
                elif movement == 3: jump_dir = np.array([-1.0, 0.0]) # Left
                elif movement == 4: jump_dir = np.array([1.0, 0.0]) # Right
                jump_executed = True

            if jump_executed:
                self.player_vel[0] = jump_dir[0] * self.JUMP_XY_VEL
                self.player_vel[1] = jump_dir[1] * self.JUMP_XY_VEL
                self.player_vel[2] = self.JUMP_Z_VEL
                self.player_on_ground = False
                self.player_jump_cooldown = 10 # Prevent mashing
                # sfx: jump
                self._create_particles(self.player_pos, 10, (200, 200, 200), 1, 15)

        if not jump_executed:
            reward -= 0.01 # No-op penalty

        # Player physics
        if not self.player_on_ground:
            self.player_vel[2] += self.GRAVITY
            self.player_pos += self.player_vel
            # Dampen horizontal movement
            self.player_vel[0] *= 0.95
            self.player_vel[1] *= 0.95
        
        # Horizontal boundary checks
        if self.player_pos[0] < 0: self.player_pos[0] = 0
        if self.player_pos[0] > self.WORLD_WIDTH: self.player_pos[0] = self.WORLD_WIDTH

        # Vertical movement reward
        y_change = self.player_pos[1] - old_y_pos
        if y_change > 0:
            reward += 0.1 * y_change
        elif y_change < 0:
            reward -= 0.2 * abs(y_change)
        
        self.score += reward

        # Ground/Landing check
        is_supported = False
        if self.player_vel[2] <= 0:  # Only check for support if not moving up
            for plat in self.platforms:
                surface_z = plat['pos'][2] + plat['depth']
                is_within_x = plat['pos'][0] <= self.player_pos[0] <= plat['pos'][0] + plat['w']
                is_within_y = plat['pos'][1] <= self.player_pos[1] <= plat['pos'][1] + plat['h']

                # Check if player is on or has just passed through the surface.
                # A tolerance window handles both standing still and falling.
                if is_within_x and is_within_y and self.player_pos[2] <= surface_z and self.player_pos[2] > surface_z - 10.0:
                    
                    # This is a new landing event (was not on ground last frame)
                    if not self.player_on_ground:
                        self._create_particles(self.player_pos, 15, plat['color'], 1.5, 20)
                        if plat['type'] == 'green':
                            reward += 1
                        elif plat['type'] == 'red':
                            reward += 5
                        elif plat['type'] == 'top':
                            reward += 100
                            self.score += 100
                            self.game_over = True

                    # Update player state for being on the ground
                    is_supported = True
                    self.player_pos[2] = surface_z  # Snap to surface
                    self.player_vel = np.array([0.0, 0.0, 0.0])
                    break  # Collision resolved

        self.player_on_ground = is_supported
        
        # Enemy movement and collision
        for enemy in self.enemies:
            platform = self.platforms[enemy['platform_idx']]
            enemy['pos'][0] += enemy['dir'] * self.enemy_speed
            if not (platform['pos'][0] <= enemy['pos'][0] <= platform['pos'][0] + platform['w'] - enemy['size']):
                enemy['dir'] *= -1
                enemy['pos'][0] += enemy['dir'] * self.enemy_speed

            # Player-enemy collision
            dist = np.linalg.norm(self.player_pos - enemy['pos'])
            if dist < enemy['size'] + 1.5:
                reward -= 20
                self.score -= 20
                self.game_over = True
                # sfx: player_hit
                self._create_particles(self.player_pos, 30, self.COLOR_ENEMY, 3, 30)
                break
        
        # Camera scrolling
        if self.player_pos[1] * self.ISO_TILE_HEIGHT_HALF - self.camera_y > self.HEIGHT * 0.7:
            self.camera_y = self.player_pos[1] * self.ISO_TILE_HEIGHT_HALF - self.HEIGHT * 0.7

        # Termination checks
        terminated = self.game_over
        if not self.player_on_ground and self.player_pos[2] < -100: # Fell off
            reward -= 50
            self.score -= 50
            terminated = True
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.enemy_speed = min(2.0, self.enemy_speed + 0.05)
        if self.steps > 0 and self.steps % 500 == 0:
            self.platform_spacing = min(50, self.platform_spacing + 5)

        # Particle update
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'][2] += self.GRAVITY * 0.5
            p['life'] -= 1

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_game(self):
        # Draw grid
        for i in range(0, self.WORLD_WIDTH, 4):
            pygame.draw.line(self.screen, self.COLOR_GRID, self._iso_to_screen(i, -10, 0), self._iso_to_screen(i, self.WORLD_HEIGHT, 0))
        for i in range(0, self.WORLD_HEIGHT, 4):
            pygame.draw.line(self.screen, self.COLOR_GRID, self._iso_to_screen(-10, i, 0), self._iso_to_screen(self.WORLD_WIDTH, i, 0))

        # Collect and sort all renderable objects by Y-coordinate for correct isometric drawing
        renderables = []
        for plat in self.platforms:
            renderables.append(('platform', plat, plat['pos'][1]))
        for enemy in self.enemies:
            renderables.append(('enemy', enemy, enemy['pos'][1]))
        renderables.append(('player', None, self.player_pos[1]))
        
        renderables.sort(key=lambda item: item[2])
        
        # Draw player shadow first
        shadow_x, shadow_y = self._iso_to_screen(self.player_pos[0], self.player_pos[1], 0)
        shadow_radius = int(10 * max(0.2, 1 - self.player_pos[2] / 200))
        if shadow_radius > 0:
            shadow_surf = pygame.Surface((shadow_radius*2, shadow_radius*2), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surf, self.COLOR_SHADOW, (0, 0, shadow_radius*2, shadow_radius*2))
            self.screen.blit(shadow_surf, (shadow_x - shadow_radius, shadow_y - shadow_radius))

        # Draw sorted objects
        for r_type, obj, _ in renderables:
            if r_type == 'platform':
                self._draw_iso_cube(self.screen, obj['pos'][0], obj['pos'][1], obj['pos'][2], obj['w'], obj['h'], obj['depth'], obj['color'], obj['side_color'])
            elif r_type == 'enemy':
                self._draw_iso_cube(self.screen, obj['pos'][0], obj['pos'][1], obj['pos'][2], obj['size'], obj['size'], obj['size'], self.COLOR_ENEMY, self.COLOR_ENEMY_SIDE)
            elif r_type == 'player':
                size = 1.5
                squash = max(0.5, 1 - abs(self.player_vel[2]) * 0.1)
                stretch = max(0.5, 1 + self.player_vel[2] * 0.05)
                self._draw_iso_cube(self.screen, self.player_pos[0], self.player_pos[1], self.player_pos[2], size*squash, size*squash, size*stretch, self.COLOR_PLAYER, self.COLOR_PLAYER_SIDE)

        # Draw particles
        for p in self.particles:
            px, py = self._iso_to_screen(p['pos'][0], p['pos'][1], p['pos'][2])
            size = max(1, int(p['life'] / 5))
            pygame.draw.circle(self.screen, p['color'], (px, py), size)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

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
            "player_pos": self.player_pos,
            "player_on_ground": self.player_on_ground,
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    # For this to work, you must comment out the line:
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Re-enable display for direct play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Space Hopper")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        # Default action is no-op
        action = np.array([0, 0, 0])
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Combine pressed keys into a single action array
        # This allows for simultaneous key presses as intended by MultiDiscrete
        action_map = {
            pygame.K_UP: 1,
            pygame.K_DOWN: 2,
            pygame.K_LEFT: 3,
            pygame.K_RIGHT: 4,
        }
        
        # Prioritize certain keys if multiple are pressed
        if keys[pygame.K_UP]: action[0] = action_map[pygame.K_UP]
        elif keys[pygame.K_DOWN]: action[0] = action_map[pygame.K_DOWN]
        elif keys[pygame.K_LEFT]: action[0] = action_map[pygame.K_LEFT]
        elif keys[pygame.K_RIGHT]: action[0] = action_map[pygame.K_RIGHT]
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Match the intended FPS

    print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
    env.close()