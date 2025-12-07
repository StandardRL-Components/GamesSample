
# Generated: 2025-08-27T22:33:21.803164
# Source Brief: brief_03166.md
# Brief Index: 3166

        
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

    user_guide = (
        "Controls: Use ← and → to move. Hold Space to charge a jump, then release to leap. "
        "A bigger charge means a higher jump, but costs more time."
    )

    game_description = (
        "Ascend a procedural tower against the clock. Master the time-based jump mechanic "
        "to climb as high as you can before time runs out."
    )

    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    PIXELS_PER_METER = 40

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (40, 80, 120)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_OUTLINE = (200, 200, 0)
    COLOR_PLATFORM = (100, 100, 120)
    COLOR_PLATFORM_TOP = (150, 150, 170)
    COLOR_TEXT = (255, 255, 255)
    COLOR_PARTICLE = (255, 100, 0)

    # Player Physics
    PLAYER_SIZE = 20
    PLAYER_SPEED = 4.0
    GRAVITY = 0.8
    FRICTION = 0.85
    MIN_JUMP_FORCE = 10.0
    MAX_JUMP_FORCE = 20.0
    CHARGE_RATE = 0.05  # Power gained per frame holding space

    # Game Rules
    INITIAL_TIMER = 60.0
    MAX_HEIGHT_METERS = 50.0
    MAX_STEPS = 1800 # 60 seconds * 30 fps
    JUMP_TIME_COST_FACTOR = 2.0  # Seconds lost for a full-power jump

    # Procedural Generation
    PLATFORM_HEIGHT = 15
    MIN_PLATFORM_WIDTH = 60
    MAX_PLATFORM_WIDTH = 150
    MIN_GAP_X = 20
    MAX_GAP_X = 100
    MIN_GAP_Y = 40
    MAX_GAP_Y = 120


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
        
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        
        # Pre-render background for efficiency
        self.background = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp),
            )
            pygame.draw.line(self.background, color, (0, y), (self.WIDTH, y))

        self._initialize_state()
        self.validate_implementation()
    
    def _initialize_state(self):
        """Defines all state variables to avoid uninitialized attribute errors."""
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.is_charging = False
        self.charge_power = 0.0
        self.prev_space_held = False
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = self.INITIAL_TIMER
        self.height_reached_pixels = 0.0
        self.camera_y = 0.0
        self.platforms = []
        self.particles = []
        self.last_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()

        # Player state
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        
        # World state
        self.platforms = []
        start_platform = pygame.Rect(
            self.WIDTH / 2 - 75, self.HEIGHT - 30, 150, self.PLATFORM_HEIGHT
        )
        self.platforms.append(start_platform)
        self.highest_platform_y = start_platform.y
        self._generate_platforms(initial=True)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        
        if not self.game_over:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            # --- 1. Handle Input ---
            self._handle_input(movement, space_held)

            # --- 2. Update Physics ---
            prev_height = self.height_reached_pixels
            self._update_physics()
            height_gain = max(0, prev_height - self.player_pos.y)
            self.height_reached_pixels = max(self.height_reached_pixels, -self.player_pos.y)
            reward += (height_gain / self.PIXELS_PER_METER) * 0.1 # Reward for gaining height

            # --- 3. Collisions & State Updates ---
            landed_this_frame = self._handle_collisions()
            if landed_this_frame:
                reward += 1.0 # Reward for landing

            # --- 4. Update Game State ---
            self.timer -= 1.0 / self.FPS
            reward -= (1.0 / self.FPS) * 0.01 # Small penalty for time passing

            self._update_camera()
            self._manage_platforms()
            self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            if self.get_height_meters() >= self.MAX_HEIGHT_METERS:
                reward += 100.0 # Win bonus
            else:
                reward += -50.0 # Loss penalty
            self.game_over = True
        
        self.score += reward
        self.last_reward = reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held):
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_SPEED
        
        # Jump charging and execution
        if space_held and self.on_ground:
            self.is_charging = True
            self.charge_power = min(1.0, self.charge_power + self.CHARGE_RATE)
        
        if not space_held and self.is_charging:
            # Execute jump on release
            jump_force = self.MIN_JUMP_FORCE + self.charge_power * (self.MAX_JUMP_FORCE - self.MIN_JUMP_FORCE)
            self.player_vel.y = -jump_force
            self.on_ground = False
            self.is_charging = False
            
            # Apply time cost
            time_cost = self.charge_power * self.JUMP_TIME_COST_FACTOR
            self.timer -= time_cost
            
            self._create_jump_particles(self.charge_power)
            # sfx: jump.wav
            
            self.charge_power = 0.0

        self.prev_space_held = space_held

    def _update_physics(self):
        if not self.on_ground:
            self.player_vel.y += self.GRAVITY
        
        self.player_pos += self.player_vel
        self.player_vel.x *= self.FRICTION

        # Clamp horizontal position
        if self.player_pos.x < self.PLAYER_SIZE / 2:
            self.player_pos.x = self.PLAYER_SIZE / 2
            self.player_vel.x = 0
        if self.player_pos.x > self.WIDTH - self.PLAYER_SIZE / 2:
            self.player_pos.x = self.WIDTH - self.PLAYER_SIZE / 2
            self.player_vel.x = 0

    def _handle_collisions(self):
        landed = False
        player_rect = pygame.Rect(
            self.player_pos.x - self.PLAYER_SIZE / 2,
            self.player_pos.y - self.PLAYER_SIZE / 2,
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        
        if self.player_vel.y > 0: # Only check for landing if falling
            for plat in self.platforms:
                if player_rect.colliderect(plat) and player_rect.bottom < plat.bottom:
                    # Check if player was above the platform in the previous frame
                    if (player_rect.bottom - self.player_vel.y) <= plat.top:
                        self.player_pos.y = plat.top - self.PLAYER_SIZE / 2
                        self.player_vel.y = 0
                        self.player_vel.x = 0 # Stop horizontal movement on land
                        if not self.on_ground:
                            landed = True
                            # sfx: land.wav
                        self.on_ground = True
                        break
        return landed

    def _update_camera(self):
        target_camera_y = -self.player_pos.y + self.HEIGHT * 0.7
        # Smooth camera follow
        self.camera_y += (target_camera_y - self.camera_y) * 0.1

    def _manage_platforms(self):
        # Generate new platforms when camera moves up
        if -self.camera_y > self.highest_platform_y - self.HEIGHT * 1.5:
             self._generate_platforms()

        # Remove platforms far below the camera
        self.platforms = [
            p for p in self.platforms if p.y + self.camera_y > -self.PLATFORM_HEIGHT
        ]

    def _generate_platforms(self, initial=False):
        num_to_gen = 10 if initial else 5
        current_y = self.highest_platform_y
        
        last_platform = self.platforms[-1] if self.platforms else pygame.Rect(self.WIDTH/2, self.HEIGHT, 0, 0)

        for _ in range(num_to_gen):
            difficulty = min(1.0, (self.get_height_meters() / self.MAX_HEIGHT_METERS))
            
            width = self.MAX_PLATFORM_WIDTH - (self.MAX_PLATFORM_WIDTH - self.MIN_PLATFORM_WIDTH) * difficulty * self.np_random.random()
            
            gap_y = self.MIN_GAP_Y + (self.MAX_GAP_Y - self.MIN_GAP_Y) * self.np_random.random()
            
            max_x_offset = self.MAX_GAP_X + (self.MIN_GAP_X - self.MAX_GAP_X) * difficulty
            offset_x = self.np_random.uniform(-max_x_offset, max_x_offset)

            new_x = last_platform.centerx + offset_x
            new_x = np.clip(new_x, width / 2 + 10, self.WIDTH - width / 2 - 10)
            
            new_y = current_y - gap_y

            new_platform = pygame.Rect(new_x - width / 2, new_y, width, self.PLATFORM_HEIGHT)
            self.platforms.append(new_platform)
            current_y = new_y
            last_platform = new_platform
        
        self.highest_platform_y = min(p.y for p in self.platforms)

    def _create_jump_particles(self, power):
        num_particles = int(5 + 15 * power)
        for _ in range(num_particles):
            self.particles.append({
                'pos': self.player_pos.copy() + pygame.Vector2(0, self.PLAYER_SIZE/2),
                'vel': pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(0, 3)),
                'life': self.np_random.uniform(0.5, 1.0),
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1.0 / self.FPS
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_termination(self):
        height_m = self.get_height_meters()
        if self.timer <= 0 or height_m >= self.MAX_HEIGHT_METERS or self.steps >= self.MAX_STEPS:
            return True
        # Failure condition if player falls too far below generated platforms
        if self.platforms and self.player_pos.y > max(p.bottom for p in self.platforms) + self.HEIGHT:
            return True
        return False
    
    def get_height_meters(self):
        return self.height_reached_pixels / self.PIXELS_PER_METER

    def _get_observation(self):
        self.screen.blit(self.background, (0, 0))
        self._render_platforms()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_platforms(self):
        for plat in self.platforms:
            # Apply camera offset
            draw_rect = plat.move(0, self.camera_y)
            
            # Draw main body
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, draw_rect)
            # Draw lighter top surface for 3D effect
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_TOP, (draw_rect.x, draw_rect.y, draw_rect.width, 4))
    
    def _render_player(self):
        # Apply camera offset
        draw_pos = self.player_pos + pygame.Vector2(0, self.camera_y)
        
        # Player body
        player_rect = pygame.Rect(
            draw_pos.x - self.PLAYER_SIZE / 2,
            draw_pos.y - self.PLAYER_SIZE / 2,
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect.inflate(4, 4), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

        # Charge meter
        if self.is_charging and self.charge_power > 0:
            bar_width = 40
            bar_height = 8
            bar_x = draw_pos.x - bar_width / 2
            bar_y = draw_pos.y - self.PLAYER_SIZE / 2 - 20

            fill_width = bar_width * self.charge_power
            
            charge_color_start = (255, 255, 255)
            charge_color_end = (255, 0, 0)
            
            interp = self.charge_power
            fill_color = (
                int(charge_color_start[0] * (1 - interp) + charge_color_end[0] * interp),
                int(charge_color_start[1] * (1 - interp) + charge_color_end[1] * interp),
                int(charge_color_start[2] * (1 - interp) + charge_color_end[2] * interp),
            )

            pygame.draw.rect(self.screen, (0,0,0), (bar_x-1, bar_y-1, bar_width+2, bar_height+2), border_radius=3)
            pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_width, bar_height), border_radius=3)
            if fill_width > 0:
                pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, fill_width, bar_height), border_radius=3)
    
    def _render_particles(self):
        for p in self.particles:
            draw_pos = p['pos'] + pygame.Vector2(0, self.camera_y)
            alpha = int(255 * (p['life'] / 1.0))
            size = int(p['size'] * (p['life'] / 1.0))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.COLOR_PARTICLE, alpha), (size, size), size)
                self.screen.blit(s, (int(draw_pos.x - size), int(draw_pos.y - size)))

    def _render_ui(self):
        # Height display
        height_text = f"HEIGHT: {self.get_height_meters():.1f}m"
        height_surf = self.font_medium.render(height_text, True, self.COLOR_TEXT)
        self.screen.blit(height_surf, (10, 10))

        # Timer display
        timer_ratio = max(0, self.timer) / self.INITIAL_TIMER
        if timer_ratio < 0.25:
            timer_color = (255, 50, 50)
        elif timer_ratio < 0.5:
            timer_color = (255, 255, 50)
        else:
            timer_color = (50, 255, 50)
        
        timer_text = f"{max(0, self.timer):.1f}"
        timer_surf = self.font_large.render(timer_text, True, timer_color)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 10, 10))

        # Score display
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_medium.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.WIDTH / 2 - score_surf.get_width() / 2, self.HEIGHT - 40))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = "TOWER COMPLETE!" if self.get_height_meters() >= self.MAX_HEIGHT_METERS else "TIME'S UP!"
            end_surf = self.font_large.render(end_text, True, self.COLOR_TEXT)
            self.screen.blit(end_surf, (self.WIDTH/2 - end_surf.get_width()/2, self.HEIGHT/2 - end_surf.get_height()/2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height_meters": self.get_height_meters(),
            "timer": self.timer,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Jumper")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it.
        # Pygame uses (width, height), but our obs is (height, width, 3).
        # We need to transpose it back for displaying.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Height: {info['height_meters']:.1f}m")
            pygame.time.wait(3000) # Wait 3 seconds
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(env.FPS)
        
    env.close()