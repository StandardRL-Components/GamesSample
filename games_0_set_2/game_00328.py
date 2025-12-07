
# Generated: 2025-08-27T13:19:22.696963
# Source Brief: brief_00328.md
# Brief Index: 328

        
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
        "Controls: ↑↓←→ to move. Press Space to collect nearby items. Hold Shift to dash."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Explore a procedurally generated forest, collect sparkling items, and find the hidden treasure before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG_SKY = (25, 29, 45)
    COLOR_BG_TREES_FAR = (40, 48, 70)
    COLOR_BG_TREES_NEAR = (55, 65, 90)
    COLOR_GROUND = (60, 110, 70)
    COLOR_PLAYER = (255, 80, 80)
    COLOR_PLAYER_GLOW = (255, 150, 150, 50)
    COLOR_ITEM = (255, 220, 50)
    COLOR_ITEM_SPARKLE = (255, 255, 180)
    COLOR_TREASURE = (255, 215, 0)
    COLOR_TREASURE_GLOW = (255, 235, 120, 30)
    COLOR_TREE_TRUNK = (80, 60, 40)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_TEXT_WARN = (255, 50, 50)

    # Screen and World
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WORLD_WIDTH = 2500
    WORLD_HEIGHT = 400

    # Game Parameters
    FPS = 30
    MAX_TIME = 60  # seconds
    MAX_STEPS = MAX_TIME * FPS

    # Player Physics
    PLAYER_ACCEL = 0.6
    PLAYER_FRICTION = 0.92
    PLAYER_MAX_SPEED = 6.0
    PLAYER_SIZE = 16
    DASH_SPEED = 20.0
    DASH_DURATION = 5  # frames
    DASH_COOLDOWN = 30 # frames

    # Interaction
    COLLECT_RADIUS = 40
    DASH_REWARD_RADIUS = 150

    # Generation
    NUM_ITEMS = 20
    NUM_TREES = 50

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
        self.font_ui = pygame.font.SysFont("sans-serif", 24)
        self.font_ui_large = pygame.font.SysFont("sans-serif", 48, bold=True)
        
        # Initialize state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.last_movement_direction = pygame.Vector2(1, 0)
        self.camera_offset_x = 0.0
        
        self.items = []
        self.trees = []
        self.treasure = {}
        self.particles = []
        
        self.time_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.dash_timer = 0
        self.dash_cooldown_timer = 0
        
        self.last_player_dist_to_treasure = 0.0
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.time_remaining = self.MAX_STEPS

        # Player state
        self.player_pos = pygame.Vector2(100, self.WORLD_HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.last_movement_direction = pygame.Vector2(1, 0)
        self.dash_timer = 0
        self.dash_cooldown_timer = 0

        # World generation
        self._generate_world()
        
        self.last_player_dist_to_treasure = self.player_pos.distance_to(self.treasure['pos'])
        
        self.particles = []

        return self._get_observation(), self._get_info()

    def _generate_world(self):
        # Ground level
        ground_y = self.WORLD_HEIGHT - 30

        # Treasure
        treasure_x = self.np_random.uniform(self.WORLD_WIDTH * 0.8, self.WORLD_WIDTH - 50)
        treasure_y = self.np_random.uniform(ground_y - 100, ground_y - 20)
        self.treasure = {
            'pos': pygame.Vector2(treasure_x, treasure_y),
            'rect': pygame.Rect(treasure_x - 12, treasure_y - 12, 24, 24)
        }
        
        # Trees (Obstacles)
        self.trees = []
        for _ in range(self.NUM_TREES):
            width = self.np_random.integers(15, 30)
            height = self.np_random.integers(50, 150)
            pos_x = self.np_random.uniform(200, self.WORLD_WIDTH - 50)
            pos_y = ground_y - height + 10 # Trunk slightly in ground
            rect = pygame.Rect(pos_x - width / 2, pos_y, width, height)
            
            # Ensure no overlap with treasure
            if not rect.colliderect(self.treasure['rect']):
                self.trees.append({'rect': rect, 'height': height})
        
        # Items (Collectibles)
        self.items = []
        for _ in range(self.NUM_ITEMS):
            is_valid_pos = False
            while not is_valid_pos:
                pos = pygame.Vector2(
                    self.np_random.uniform(200, self.WORLD_WIDTH - 50),
                    self.np_random.uniform(100, ground_y - 20)
                )
                
                # Check for overlap with trees
                is_in_tree = any(tree['rect'].collidepoint(pos) for tree in self.trees)
                
                if not is_in_tree:
                    self.items.append({
                        'pos': pos, 
                        'active': True, 
                        'sparkle_size': self.np_random.uniform(1, 4)
                    })
                    is_valid_pos = True

    def step(self, action):
        reward = -0.1  # Step penalty
        
        # --- Update Timers ---
        self.time_remaining -= 1
        if self.dash_timer > 0: self.dash_timer -= 1
        if self.dash_cooldown_timer > 0: self.dash_cooldown_timer -= 1
        
        # --- Unpack Actions ---
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # --- Handle Dash ---
        if shift_pressed and self.dash_cooldown_timer <= 0 and self.dash_timer <= 0:
            self.dash_timer = self.DASH_DURATION
            self.dash_cooldown_timer = self.DASH_COOLDOWN
            self.player_vel = self.last_movement_direction.normalize() * self.DASH_SPEED
            # sfx: dash_whoosh.wav
            
            # Dash reward logic
            is_item_nearby = any(self.player_pos.distance_to(item['pos']) < self.DASH_REWARD_RADIUS for item in self.items if item['active'])
            reward += 10 if is_item_nearby else -2
            
            # Dash particles
            for _ in range(20):
                self._create_particle(
                    self.player_pos,
                    self.COLOR_PLAYER_GLOW[:3],
                    vel=self.player_vel.rotate(self.np_random.uniform(-30, 30)) * 0.2,
                    life=15,
                    size=self.np_random.integers(2, 5)
                )
        
        # --- Handle Movement ---
        if self.dash_timer <= 0: # No movement control during dash
            accel = pygame.Vector2(0, 0)
            moved = False
            if movement == 1: accel.y = -self.PLAYER_ACCEL; moved = True
            elif movement == 2: accel.y = self.PLAYER_ACCEL; moved = True
            elif movement == 3: accel.x = -self.PLAYER_ACCEL; moved = True
            elif movement == 4: accel.x = self.PLAYER_ACCEL; moved = True

            self.player_vel += accel
            self.player_vel *= self.PLAYER_FRICTION
            
            if self.player_vel.length() > self.PLAYER_MAX_SPEED:
                self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)

            if moved:
                self.last_movement_direction = self.player_vel.copy() if self.player_vel.length() > 0.1 else self.last_movement_direction
        
        self.player_pos += self.player_vel

        # --- Handle Collisions ---
        self._handle_collisions()

        # --- Handle Item Collection ---
        if space_pressed:
            for item in self.items:
                if item['active'] and self.player_pos.distance_to(item['pos']) < self.COLLECT_RADIUS:
                    item['active'] = False
                    self.score += 1
                    reward += 5
                    # sfx: item_collect.wav
                    for _ in range(30):
                        self._create_particle(
                            item['pos'], 
                            self.COLOR_ITEM, 
                            life=20, 
                            size=self.np_random.uniform(1, 4),
                            gravity=0.1
                        )
                    break # Collect one item per press
        
        # --- Update Game Logic & State ---
        self.steps += 1
        self._update_particles()
        self._update_camera()

        # Distance to treasure reward
        current_dist = self.player_pos.distance_to(self.treasure['pos'])
        reward += (self.last_player_dist_to_treasure - current_dist) * 0.5
        self.last_player_dist_to_treasure = current_dist
        
        # --- Check Termination Conditions ---
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE/2, self.player_pos.y - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        if player_rect.colliderect(self.treasure['rect']):
            self.game_over = True
            self.game_won = True
            reward += 100
            # sfx: win_jingle.wav
        
        if self.time_remaining <= 0:
            self.game_over = True
            reward -= 10
            # sfx: lose_sound.wav
            
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_collisions(self):
        # World boundaries
        self.player_pos.x = max(self.PLAYER_SIZE/2, min(self.player_pos.x, self.WORLD_WIDTH - self.PLAYER_SIZE/2))
        self.player_pos.y = max(self.PLAYER_SIZE/2, min(self.player_pos.y, self.WORLD_HEIGHT - 30 - self.PLAYER_SIZE/2))

        # Tree collisions
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE/2, self.player_pos.y - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for tree in self.trees:
            if player_rect.colliderect(tree['rect']):
                # Find overlap
                dx = self.player_pos.x - tree['rect'].centerx
                dy = self.player_pos.y - tree['rect'].centery
                overlap_x = (player_rect.width / 2 + tree['rect'].width / 2) - abs(dx)
                overlap_y = (player_rect.height / 2 + tree['rect'].height / 2) - abs(dy)

                # Resolve collision by moving player out of the obstacle
                if overlap_x < overlap_y:
                    if dx > 0: self.player_pos.x += overlap_x
                    else: self.player_pos.x -= overlap_x
                    self.player_vel.x = 0
                else:
                    if dy > 0: self.player_pos.y += overlap_y
                    else: self.player_pos.y -= overlap_y
                    self.player_vel.y = 0
                # sfx: thump.wav

    def _update_camera(self):
        target_camera_x = self.player_pos.x - self.SCREEN_WIDTH / 2
        # Smooth camera follow using linear interpolation (lerp)
        self.camera_offset_x += (target_camera_x - self.camera_offset_x) * 0.1
        # Clamp camera to world bounds
        self.camera_offset_x = max(0, min(self.camera_offset_x, self.WORLD_WIDTH - self.SCREEN_WIDTH))

    def _create_particle(self, pos, color, vel=None, life=30, size=3, gravity=0):
        if vel is None:
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
        
        self.particles.append({
            'pos': pos.copy(),
            'vel': vel.copy(),
            'life': life,
            'max_life': life,
            'size': size,
            'color': color,
            'gravity': gravity
        })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'].y += p['gravity']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
    
    def _get_observation(self):
        # --- Render Background ---
        self.screen.fill(self.COLOR_BG_SKY)
        
        # Parallax background trees
        for i in range(20):
            x = (i * 150 - self.camera_offset_x * 0.3) % (self.WORLD_WIDTH * 1.2) - 100
            pygame.draw.rect(self.screen, self.COLOR_BG_TREES_FAR, (int(x), 150, 60, 200))
        for i in range(15):
            x = (i * 200 - self.camera_offset_x * 0.6) % (self.WORLD_WIDTH * 1.1) - 100
            pygame.draw.rect(self.screen, self.COLOR_BG_TREES_NEAR, (int(x), 100, 80, 250))

        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.WORLD_HEIGHT - 30, self.SCREEN_WIDTH, 30))
        
        # --- Render Game Elements (relative to camera) ---
        cam_x = int(self.camera_offset_x)

        # Trees
        for tree in self.trees:
            trunk = tree['rect']
            canopy_rect = pygame.Rect(trunk.x - 10, trunk.y - tree['height']*0.5, trunk.width + 20, tree['height'])
            pygame.draw.rect(self.screen, self.COLOR_TREE_TRUNK, (trunk.x - cam_x, trunk.y, trunk.width, trunk.height))
            pygame.draw.ellipse(self.screen, self.COLOR_GROUND, (canopy_rect.x - cam_x, canopy_rect.y, canopy_rect.width, canopy_rect.height))

        # Items
        for item in self.items:
            if item['active']:
                pos_on_screen = (int(item['pos'].x - cam_x), int(item['pos'].y))
                pygame.gfxdraw.filled_circle(self.screen, pos_on_screen[0], pos_on_screen[1], 6, self.COLOR_ITEM)
                
                # Sparkle animation
                sparkle_rad = abs(math.sin(self.steps * 0.2 + item['sparkle_size'])) * 3 + 2
                pygame.gfxdraw.filled_circle(self.screen, pos_on_screen[0], pos_on_screen[1], int(sparkle_rad), self.COLOR_ITEM_SPARKLE)
        
        # Treasure
        pos_on_screen = (int(self.treasure['pos'].x - cam_x), int(self.treasure['pos'].y))
        glow_size = 15 + abs(math.sin(self.steps * 0.05)) * 10
        pygame.gfxdraw.filled_circle(self.screen, pos_on_screen[0], pos_on_screen[1], int(glow_size), self.COLOR_TREASURE_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, pos_on_screen[0], pos_on_screen[1], 12, self.COLOR_TREASURE)
        pygame.gfxdraw.filled_circle(self.screen, pos_on_screen[0], pos_on_screen[1], 8, self.COLOR_ITEM_SPARKLE)

        # Particles
        for p in self.particles:
            pos = (int(p['pos'].x - cam_x), int(p['pos'].y))
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (pos[0] - size, pos[1] - size))

        # Player
        player_screen_pos = (int(self.player_pos.x - cam_x), int(self.player_pos.y))
        size = int(self.PLAYER_SIZE)
        
        # Player glow/dash effect
        if self.dash_timer > 0:
            glow_rad = size * (2.0 + (self.DASH_DURATION - self.dash_timer) / self.DASH_DURATION)
            pygame.gfxdraw.filled_circle(self.screen, player_screen_pos[0], player_screen_pos[1], int(glow_rad), self.COLOR_PLAYER_GLOW)
        
        # Player body with bobbing animation
        bob = math.sin(self.steps * 0.2) * 2
        player_rect = pygame.Rect(player_screen_pos[0] - size/2, player_screen_pos[1] - size/2 + bob, size, size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # --- Render UI ---
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_ui(self):
        # Time display
        time_sec = self.time_remaining // self.FPS
        time_text = f"Time: {time_sec}"
        time_color = self.COLOR_UI_TEXT if time_sec > 10 else self.COLOR_UI_TEXT_WARN
        time_surf = self.font_ui.render(time_text, True, time_color)
        self.screen.blit(time_surf, (10, 10))
        
        # Score display
        score_text = f"Items: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 35))

        # Dash Cooldown
        if self.dash_cooldown_timer > 0:
            bar_width = 100
            fill_width = int(bar_width * (1 - self.dash_cooldown_timer / self.DASH_COOLDOWN))
            pygame.draw.rect(self.screen, (50, 50, 80), (self.SCREEN_WIDTH - bar_width - 10, 10, bar_width, 20))
            pygame.draw.rect(self.screen, (100, 100, 200), (self.SCREEN_WIDTH - bar_width - 10, 10, fill_width, 20))
            dash_text = self.font_ui.render("DASH", True, self.COLOR_UI_TEXT)
            self.screen.blit(dash_text, (self.SCREEN_WIDTH - bar_width - 70, 8))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "TREASURE FOUND!" if self.game_won else "TIME UP!"
            color = self.COLOR_TREASURE if self.game_won else self.COLOR_UI_TEXT_WARN
            msg_surf = self.font_ui_large.render(message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining // self.FPS,
            "player_pos": (self.player_pos.x, self.player_pos.y),
            "treasure_pos": (self.treasure['pos'].x, self.treasure['pos'].y)
        }
    
    def close(self):
        pygame.quit()

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set SDL_VIDEODRIVER to "dummy" for headless execution
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # Pygame setup for display
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Forest Treasure Hunt")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    terminated = False
    
    print("\n" + "="*30)
    print("      FOREST TREASURE HUNT")
    print("="*30)
    print(env.game_description)
    print("\n" + env.user_guide + "\n")

    while not terminated:
        # --- Get Player Input ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_pressed = 1 if keys[pygame.K_SPACE] else 0
        shift_pressed = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_pressed, shift_pressed]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to Display ---
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # And the axes are transposed from pygame's internal representation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Handle Quit Event ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # --- Control Framerate ---
        clock.tick(GameEnv.FPS)

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a moment before closing
            pygame.time.wait(3000)

    env.close()