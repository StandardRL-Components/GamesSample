
# Generated: 2025-08-27T16:34:57.276833
# Source Brief: brief_01268.md
# Brief Index: 1268

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to aim your jump. Hold Space for a long jump or Shift for a short jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop across procedurally generated platforms, dodging obstacles, to reach the end of the level."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PLAYER_SIZE = 20
    GRAVITY = 0.4
    JUMP_POWER_SHORT = -6.0
    JUMP_POWER_LONG = -9.0
    JUMP_HORIZONTAL_SPEED = 4.0
    MAX_STEPS = 2000
    PLATFORM_COUNT_GOAL = 30

    # --- Colors ---
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (0, 0, 10)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (150, 255, 150, 50)
    COLOR_PLATFORM = (150, 150, 150)
    COLOR_PLATFORM_OUTLINE = (220, 220, 220)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (255, 150, 150, 100)
    COLOR_PARTICLE = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_FINAL_PLATFORM = (255, 215, 0)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_on_platform = False
        self.last_jump_direction = None
        
        self.platforms = []
        self.obstacles = []
        self.particles = []
        
        self.camera_y = 0
        self.highest_platform_y = self.HEIGHT - 50
        self.next_platform_idx = 0
        self.visited_platforms = set()

        self.base_obstacle_speed = 1.0
        self.base_platform_gap = 100

        self.rng = None
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            # If no seed, create a new generator
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_on_platform = True
        self.last_jump_direction = pygame.Vector2(0, -1) # Default to up

        self.platforms = []
        self.obstacles = []
        self.particles = []

        # Difficulty scaling parameters
        self.current_obstacle_speed = 1.0
        self.current_platform_gap = 100

        self.camera_y = 0
        self.highest_platform_y = self.HEIGHT - 50
        self.next_platform_idx = 0
        self.visited_platforms = set()
        
        # Create initial platforms
        initial_platform = pygame.Rect(self.WIDTH/2 - 50, self.HEIGHT - 30, 100, 20)
        self.platforms.append({'rect': initial_platform, 'id': self.next_platform_idx, 'type': 'normal'})
        self.next_platform_idx += 1
        self.visited_platforms.add(0)

        for _ in range(15): # Pre-generate some platforms
            self._generate_platform()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean for long jump
        shift_held = action[2] == 1  # Boolean for short jump
        
        # --- Handle Input and Player Logic ---
        self._handle_input(movement, space_held, shift_held)
        
        # --- Update Game Physics ---
        self._update_player_physics()
        
        # --- Check Collisions and Update State ---
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- Update Dynamic Elements ---
        self._update_obstacles()
        self._update_particles()
        
        # --- Procedural Generation & Difficulty ---
        self._manage_world()
        self._update_difficulty()
        
        # --- Calculate Final Reward for Step ---
        reward += 0.1  # Survival reward

        # --- Check Termination Conditions ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.game_over and self.lives <= 0:
            reward -= 50 # Penalty for losing all lives (in addition to per-life penalty)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Update aiming direction
        if movement == 1: self.last_jump_direction = pygame.Vector2(0, -1) # Up
        elif movement == 2: self.last_jump_direction = pygame.Vector2(0, 1) # Down
        elif movement == 3: self.last_jump_direction = pygame.Vector2(-1, 0) # Left
        elif movement == 4: self.last_jump_direction = pygame.Vector2(1, 0) # Right
        
        # Execute jump if on a platform and a jump key is pressed
        if self.player_on_platform:
            jump_power = 0
            if space_held: # Long jump
                jump_power = self.JUMP_POWER_LONG
            elif shift_held: # Short jump
                jump_power = self.JUMP_POWER_SHORT

            if jump_power != 0:
                self.player_on_platform = False
                # Sound: Player Jump
                jump_vector = self.last_jump_direction.normalize()
                self.player_vel.y = jump_vector.y * jump_power
                self.player_vel.x = jump_vector.x * abs(jump_power) * 0.8 # Horizontal speed related to jump power
                # Small upward boost for horizontal jumps
                if abs(jump_vector.x) > 0.1:
                    self.player_vel.y = min(self.player_vel.y, self.JUMP_POWER_SHORT * 0.8)


    def _update_player_physics(self):
        if not self.player_on_platform:
            self.player_vel.y += self.GRAVITY
        
        self.player_pos += self.player_vel
        
        # Prevent player from going off sides of screen
        if self.player_pos.x < 0:
            self.player_pos.x = 0
            self.player_vel.x = 0
        if self.player_pos.x > self.WIDTH - self.PLAYER_SIZE:
            self.player_pos.x = self.WIDTH - self.PLAYER_SIZE
            self.player_vel.x = 0

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Player vs. Platforms
        if self.player_vel.y > 0: # Only check for landing if falling
            self.player_on_platform = False
            for p in self.platforms:
                if p['rect'].colliderect(player_rect) and player_rect.bottom < p['rect'].centery:
                    self.player_pos.y = p['rect'].top - self.PLAYER_SIZE
                    self.player_vel.y = 0
                    self.player_vel.x = 0
                    self.player_on_platform = True
                    # Sound: Player Land
                    self._create_particles(pygame.Vector2(player_rect.centerx, p['rect'].top))
                    
                    if p['id'] not in self.visited_platforms:
                        self.visited_platforms.add(p['id'])
                        self.score += 10
                        reward += 5.0 # Reward for reaching a new platform
                        
                        if p['type'] == 'final':
                            self.score += 500
                            reward += 100.0
                            self.game_over = True
                    break
        
        # Player vs. Obstacles
        for o in self.obstacles:
            obstacle_pos = pygame.Vector2(o['rect'].center)
            player_center = pygame.Vector2(player_rect.center)
            if obstacle_pos.distance_to(player_center) < o['rect'].width / 2 + self.PLAYER_SIZE / 2:
                reward -= 1.0 # Penalty for hitting obstacle
                # Sound: Player Hit
                self._lose_life()
                break

        # Player vs. Fall
        if self.player_pos.y > self.camera_y + self.HEIGHT:
            self._lose_life()
            
        return reward
        
    def _lose_life(self):
        self.lives -= 1
        self.score = max(0, self.score - 100)
        # Sound: Life Lost
        if self.lives <= 0:
            self.game_over = True
        else:
            # Respawn at the last visited platform
            last_platform = max(self.visited_platforms) if self.visited_platforms else 0
            for p in self.platforms:
                if p['id'] == last_platform:
                    self.player_pos = pygame.Vector2(p['rect'].centerx - self.PLAYER_SIZE/2, p['rect'].top - self.PLAYER_SIZE)
                    self.player_vel = pygame.Vector2(0, 0)
                    self.player_on_platform = True
                    break

    def _update_obstacles(self):
        for o in self.obstacles:
            o['rect'].x += o['vel_x']
            if o['rect'].left < o['platform_rect'].left or o['rect'].right > o['platform_rect'].right:
                o['vel_x'] *= -1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)

    def _manage_world(self):
        # Smooth camera follow
        target_camera_y = self.player_pos.y - self.HEIGHT * 0.6
        self.camera_y += (target_camera_y - self.camera_y) * 0.05

        # Generate new platforms when player reaches a certain height
        while self.highest_platform_y > self.camera_y - 50:
             self._generate_platform()

        # Remove off-screen elements
        self.platforms = [p for p in self.platforms if p['rect'].bottom > self.camera_y]
        self.obstacles = [o for o in self.obstacles if o['rect'].bottom > self.camera_y]

    def _generate_platform(self):
        last_platform = self.platforms[-1]['rect']
        
        is_final = self.next_platform_idx == self.PLATFORM_COUNT_GOAL
        
        if is_final:
            width = 200
            height = 40
            x = self.WIDTH / 2 - width / 2
            y = last_platform.y - 150
            ptype = 'final'
        else:
            width = self.rng.integers(60, 150)
            height = 20
            
            angle = self.rng.uniform(-math.pi * 0.6, math.pi * 0.6)
            distance = self.rng.uniform(self.current_platform_gap * 0.8, self.current_platform_gap * 1.2)
            
            x_offset = math.sin(angle) * distance
            y_offset = math.cos(angle) * distance
            
            x = last_platform.centerx + x_offset - width / 2
            y = last_platform.y - y_offset
            
            x = np.clip(x, 20, self.WIDTH - width - 20)
            ptype = 'normal'

        new_platform_rect = pygame.Rect(x, y, width, height)
        new_platform = {'rect': new_platform_rect, 'id': self.next_platform_idx, 'type': ptype}
        self.platforms.append(new_platform)
        
        self.highest_platform_y = min(self.highest_platform_y, y)
        self.next_platform_idx += 1
        
        # Occasionally add an obstacle
        if not is_final and self.rng.random() < 0.3 and self.steps > 100:
            ox = new_platform_rect.centerx
            oy = new_platform_rect.top - 10
            radius = 10
            obstacle_rect = pygame.Rect(ox - radius, oy - radius, radius*2, radius*2)
            vel_x = (self.rng.choice([-1, 1])) * self.current_obstacle_speed
            self.obstacles.append({'rect': obstacle_rect, 'platform_rect': new_platform_rect, 'vel_x': vel_x})


    def _update_difficulty(self):
        # Obstacle speed increases by 0.05 pixels/frame every 200 steps.
        self.current_obstacle_speed = self.base_obstacle_speed + 0.05 * (self.steps // 200)
        # Platform gaps increase by 5 pixels every 500 steps.
        self.current_platform_gap = self.base_platform_gap + 5 * (self.steps // 500)

    def _create_particles(self, pos, count=10):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 3)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': self.rng.uniform(2, 5),
                'lifespan': self.rng.integers(20, 40)
            })

    def _get_observation(self):
        # --- Render Background ---
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        # --- Render Game Elements (with camera offset) ---
        self._render_platforms()
        self._render_obstacles()
        self._render_particles()
        self._render_player()
        
        # --- Render UI (no camera offset) ---
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_platforms(self):
        for p in self.platforms:
            r = p['rect']
            cam_rect = r.move(0, -self.camera_y)
            color = self.COLOR_FINAL_PLATFORM if p['type'] == 'final' else self.COLOR_PLATFORM
            pygame.draw.rect(self.screen, color, cam_rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, cam_rect, 2, border_radius=4)

    def _render_obstacles(self):
        for o in self.obstacles:
            r = o['rect']
            pos = (int(r.centerx), int(r.centery - self.camera_y))
            radius = int(r.width / 2)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_OBSTACLE)
            # Glow effect
            s = pygame.Surface((radius*4, radius*4), pygame.SRCALPHA)
            pygame.draw.circle(s, self.COLOR_OBSTACLE_GLOW, (radius*2, radius*2), radius*1.5)
            self.screen.blit(s, (pos[0] - radius*2, pos[1] - radius*2), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y - self.camera_y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), self.COLOR_PARTICLE)

    def _render_player(self):
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y - self.camera_y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Glow effect
        s = pygame.Surface((self.PLAYER_SIZE*4, self.PLAYER_SIZE*4), pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_PLAYER_GLOW, (self.PLAYER_SIZE*1.5, self.PLAYER_SIZE*1.5, self.PLAYER_SIZE, self.PLAYER_SIZE), border_radius=8)
        self.screen.blit(s, (player_rect.x - self.PLAYER_SIZE*1.5, player_rect.y - self.PLAYER_SIZE*1.5), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
    
    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        heart_size = 20
        for i in range(self.lives):
            x = self.WIDTH - (i + 1) * (heart_size + 5) - 5
            self._draw_heart(x, 15, heart_size)
            
        if self.game_over:
            msg = "GOAL REACHED!" if self.lives > 0 else "GAME OVER"
            text_surf = self.font.render(msg, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20,20))
            self.screen.blit(text_surf, text_rect)

    def _draw_heart(self, x, y, size):
        points = [
            (x, y - size * 0.25),
            (x - size * 0.5, y - size * 0.5),
            (x - size * 0.5, y),
            (x, y + size * 0.5),
            (x + size * 0.5, y),
            (x + size * 0.5, y - size * 0.5),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_OBSTACLE)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_OBSTACLE)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "platforms_visited": len(self.visited_platforms),
        }
    
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
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Space Hopper")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Transpose observation back for Pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for reset
            pass
            
        clock.tick(30) # Run at 30 FPS
        
    env.close()