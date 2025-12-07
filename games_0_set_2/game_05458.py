
# Generated: 2025-08-28T05:04:56.515500
# Source Brief: brief_05458.md
# Brief Index: 5458

        
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

    user_guide = (
        "Controls: ↑ to accelerate up, ↓ to accelerate down. Avoid the cave walls and collect gems!"
    )

    game_description = (
        "Steer a mine cart through a procedurally generated cave. Collect 50 gems to win, but be careful not to crash into the walls. The deeper you go, the faster it gets!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 2500 # Increased for longer gameplay potential
        self.VICTORY_GEMS = 50

        # Colors (Bright interactive, darker background)
        self.COLOR_BG_DARK = (25, 20, 20)
        self.COLOR_BG_STREAK = (45, 40, 40)
        self.COLOR_WALL = (80, 70, 70)
        self.COLOR_WALL_HIGHLIGHT = (100, 90, 90)
        self.COLOR_CART_BODY = (139, 69, 19) # Brown
        self.COLOR_CART_TRIM = (90, 50, 10)
        self.COLOR_GEM = (255, 230, 0)
        self.COLOR_GEM_SPARKLE = (255, 255, 150)
        self.COLOR_DUST = (115, 100, 80)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_BG = (10, 10, 10, 180)

        # Player Physics
        self.CART_X_POS = 100
        self.CART_WIDTH = 40
        self.CART_HEIGHT = 28
        self.CART_ACCEL = 0.5
        self.CART_DRAG = 0.95
        self.MAX_VY = 8

        # World Generation
        self.WALL_SEGMENT_WIDTH = 80
        self.WALL_MIN_GAP = 120
        self.WALL_MAX_GAP = 200
        self.WALL_MAX_Y_CHANGE = 60
        
        # Rewards
        self.REWARD_SURVIVAL = 0.01
        self.REWARD_GEM = 1.0
        self.REWARD_NEAR_WALL = -0.1 # Less punitive than brief for smoother learning
        self.REWARD_WIN = 50.0
        self.REWARD_WALL_HIT = -50.0
        self.NEAR_WALL_DISTANCE = 15

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_game_over = pygame.font.SysFont(None, 60)

        # --- Initialize State ---
        self.np_random = None
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.gem_count = 0
        self.game_over = False
        self.win = False

        self.cart = {
            "rect": pygame.Rect(self.CART_X_POS, self.SCREEN_HEIGHT / 2, self.CART_WIDTH, self.CART_HEIGHT),
            "vy": 0,
        }

        self.scroll_speed = 2.0
        self.walls = []
        self.gems = []
        self.particles = []
        self.bg_streaks = []

        self._generate_initial_world()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(60) # Higher tick rate for smoother physics

        reward = self.REWARD_SURVIVAL

        if not self.game_over:
            self._handle_input(action)
            self._update_player()
            self._update_world()

            reward += self._handle_gem_collection()
            
            is_near_wall, did_collide = self._handle_wall_collisions()
            if is_near_wall:
                reward += self.REWARD_NEAR_WALL
                self._spawn_dust_particles()
            if did_collide:
                self.game_over = True
                # sfx: crash_sound
                reward += self.REWARD_WALL_HIT

        self._update_particles()
        
        self.steps += 1
        self.win = self.gem_count >= self.VICTORY_GEMS
        timeout = self.steps >= self.MAX_STEPS
        terminated = self.game_over or self.win or timeout

        if self.win and not self.game_over:
            # sfx: victory_fanfare
            reward += self.REWARD_WIN
            # Make sure we only give win reward once
            self.game_over = True 

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement = action[0]
        if movement == 1:  # Up
            self.cart["vy"] -= self.CART_ACCEL
        elif movement == 2:  # Down
            self.cart["vy"] += self.CART_ACCEL

    def _update_player(self):
        # Apply drag
        self.cart["vy"] *= self.CART_DRAG
        # Clamp velocity
        self.cart["vy"] = np.clip(self.cart["vy"], -self.MAX_VY, self.MAX_VY)
        # Update position
        self.cart["rect"].y += self.cart["vy"]
        # Clamp position to screen bounds
        self.cart["rect"].y = np.clip(self.cart["rect"].y, 0, self.SCREEN_HEIGHT - self.cart["rect"].height)

    def _update_world(self):
        # Scroll walls
        for wall in self.walls:
            wall["x"] -= self.scroll_speed
        
        # Scroll gems
        for gem in self.gems:
            gem["rect"].x -= self.scroll_speed

        # Scroll background streaks
        for streak in self.bg_streaks:
            streak["x"] -= self.scroll_speed * streak["speed_mod"]
            if streak["x"] + streak["w"] < 0:
                streak["x"] = self.SCREEN_WIDTH

        # Remove off-screen elements
        self.walls = [w for w in self.walls if w["x"] + self.WALL_SEGMENT_WIDTH > 0]
        self.gems = [g for g in self.gems if g["rect"].right > 0]

        # Generate new content if needed
        if not self.walls or self.walls[-1]["x"] < self.SCREEN_WIDTH - self.WALL_SEGMENT_WIDTH:
            self._generate_new_segment()

    def _generate_initial_world(self):
        # Generate background streaks
        for _ in range(50):
            self.bg_streaks.append({
                "x": self.np_random.integers(0, self.SCREEN_WIDTH),
                "y": self.np_random.integers(0, self.SCREEN_HEIGHT),
                "w": self.np_random.integers(20, 80),
                "h": self.np_random.integers(1, 3),
                "speed_mod": self.np_random.uniform(0.3, 0.6)
            })

        # Generate initial wall and gem segments
        while len(self.walls) * self.WALL_SEGMENT_WIDTH < self.SCREEN_WIDTH + self.WALL_SEGMENT_WIDTH:
             self._generate_new_segment()

    def _generate_new_segment(self):
        last_x = self.walls[-1]["x"] if self.walls else -self.WALL_SEGMENT_WIDTH
        last_gap_y = self.walls[-1]["gap_y"] if self.walls else self.SCREEN_HEIGHT / 2
        
        new_x = last_x + self.WALL_SEGMENT_WIDTH
        
        y_change = self.np_random.integers(-self.WALL_MAX_Y_CHANGE, self.WALL_MAX_Y_CHANGE + 1)
        new_gap_y = np.clip(last_gap_y + y_change, self.WALL_MAX_GAP / 2, self.SCREEN_HEIGHT - self.WALL_MAX_GAP / 2)
        
        new_gap_height = self.np_random.integers(self.WALL_MIN_GAP, self.WALL_MAX_GAP + 1)

        self.walls.append({"x": new_x, "gap_y": new_gap_y, "gap_height": new_gap_height})

        # Add gems in the new gap
        num_gems = self.np_random.integers(1, 4)
        for i in range(num_gems):
            gem_x = new_x + self.WALL_SEGMENT_WIDTH / 2 + self.np_random.integers(-10, 11)
            gem_y_offset = (i - (num_gems - 1) / 2) * 30
            gem_y = new_gap_y + gem_y_offset
            
            gem_rect = pygame.Rect(gem_x - 8, gem_y - 8, 16, 16)
            if gem_rect.top > new_gap_y - new_gap_height/2 + 10 and gem_rect.bottom < new_gap_y + new_gap_height/2 - 10:
                self.gems.append({
                    "rect": gem_rect,
                    "spawn_time": self.steps
                })

    def _handle_gem_collection(self):
        collected_reward = 0
        gems_collected_this_frame = 0
        for gem in self.gems[:]:
            if self.cart["rect"].colliderect(gem["rect"]):
                self.gems.remove(gem)
                # sfx: gem_collect_sound
                self.gem_count += 1
                gems_collected_this_frame += 1
                collected_reward += self.REWARD_GEM
                
                # Speed up every 5 gems
                if self.gem_count > 0 and self.gem_count % 5 == 0:
                    self.scroll_speed += 0.2
        return collected_reward

    def _handle_wall_collisions(self):
        is_near = False
        did_collide = False
        cart_rect = self.cart["rect"]

        for wall in self.walls:
            if wall["x"] < cart_rect.right and wall["x"] + self.WALL_SEGMENT_WIDTH > cart_rect.left:
                gap_top = wall["gap_y"] - wall["gap_height"] / 2
                gap_bottom = wall["gap_y"] + wall["gap_height"] / 2
                
                # Create rects for upper and lower wall parts
                upper_wall_rect = pygame.Rect(wall["x"], 0, self.WALL_SEGMENT_WIDTH, gap_top)
                lower_wall_rect = pygame.Rect(wall["x"], gap_bottom, self.WALL_SEGMENT_WIDTH, self.SCREEN_HEIGHT - gap_bottom)

                # Check for collision
                if cart_rect.colliderect(upper_wall_rect) or cart_rect.colliderect(lower_wall_rect):
                    did_collide = True
                    break

                # Check for proximity
                if not is_near:
                    if (0 < (cart_rect.top - upper_wall_rect.bottom) < self.NEAR_WALL_DISTANCE or
                        0 < (lower_wall_rect.top - cart_rect.bottom) < self.NEAR_WALL_DISTANCE):
                        is_near = True
        
        return is_near, did_collide
        
    def _spawn_dust_particles(self):
        # sfx: scraping_sound
        for _ in range(2):
            self.particles.append({
                'x': self.cart['rect'].left,
                'y': self.np_random.choice([self.cart['rect'].top, self.cart['rect'].bottom]),
                'vx': -self.scroll_speed * self.np_random.uniform(0.5, 1.0),
                'vy': self.np_random.uniform(-0.5, 0.5),
                'radius': self.np_random.integers(2, 5),
                'lifetime': self.np_random.integers(15, 30),
                'color': self.COLOR_DUST
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG_DARK)
        self._render_background()
        self._render_walls()
        self._render_gems()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems": self.gem_count,
            "speed": self.scroll_speed
        }

    def _render_background(self):
        for streak in self.bg_streaks:
            pygame.draw.rect(self.screen, self.COLOR_BG_STREAK, (streak["x"], streak["y"], streak["w"], streak["h"]))

    def _render_walls(self):
        for wall in self.walls:
            gap_top = wall["gap_y"] - wall["gap_height"] / 2
            gap_bottom = wall["gap_y"] + wall["gap_height"] / 2
            
            # Upper wall
            pygame.draw.rect(self.screen, self.COLOR_WALL, (wall["x"], 0, self.WALL_SEGMENT_WIDTH, gap_top))
            pygame.draw.rect(self.screen, self.COLOR_WALL_HIGHLIGHT, (wall["x"], gap_top-2, self.WALL_SEGMENT_WIDTH, 2))
            
            # Lower wall
            pygame.draw.rect(self.screen, self.COLOR_WALL, (wall["x"], gap_bottom, self.WALL_SEGMENT_WIDTH, self.SCREEN_HEIGHT - gap_bottom))
            pygame.draw.rect(self.screen, self.COLOR_WALL_HIGHLIGHT, (wall["x"], gap_bottom, self.WALL_SEGMENT_WIDTH, 2))

    def _render_gems(self):
        for gem in self.gems:
            t = (self.steps - gem["spawn_time"]) * 0.2
            size_anim = abs(math.sin(t)) * 2
            
            center_x, center_y = gem["rect"].center
            
            # Draw diamond shape
            points = [
                (center_x, center_y - 8 - size_anim),
                (center_x + 8 + size_anim, center_y),
                (center_x, center_y + 8 + size_anim),
                (center_x - 8 - size_anim, center_y)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GEM_SPARKLE)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GEM)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 30))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['x'] - p['radius']), int(p['y'] - p['radius'])))

    def _render_player(self):
        rect = self.cart["rect"]
        # Body
        body_rect = pygame.Rect(rect.left, rect.top + 5, rect.width, rect.height - 10)
        pygame.draw.rect(self.screen, self.COLOR_CART_BODY, body_rect)
        pygame.draw.rect(self.screen, self.COLOR_CART_TRIM, body_rect, 2)
        
        # Wheels
        wheel_radius = 6
        pygame.draw.circle(self.screen, self.COLOR_CART_TRIM, (rect.left + 10, rect.bottom - 5), wheel_radius)
        pygame.draw.circle(self.screen, self.COLOR_CART_TRIM, (rect.right - 10, rect.bottom - 5), wheel_radius)
        
        # Highlight
        highlight_rect = pygame.Rect(rect.left + 2, rect.top + 7, rect.width - 4, 3)
        pygame.draw.rect(self.screen, (200, 150, 100), highlight_rect)

    def _render_ui(self):
        # UI Background
        ui_surf = pygame.Surface((180, 40), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (10, 10))

        # Gem count
        gem_text = self.font_ui.render(f"GEMS: {self.gem_count}/{self.VICTORY_GEMS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(gem_text, (20, 18))

        # Game Over/Win Text
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 50, 50)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Text shadow
            shadow_text = self.font_game_over.render(msg, True, (0,0,0))
            shadow_rect = shadow_text.get_rect(center=(self.SCREEN_WIDTH / 2 + 3, self.SCREEN_HEIGHT / 2 + 3))
            self.screen.blit(shadow_text, shadow_rect)
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    obs, info = env.reset()
    
    print("Initial Observation Shape:", obs.shape)
    print("Initial Info:", info)
    
    # Test a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Reward={reward:.2f}, Terminated={terminated}, Info={info}")
        if terminated:
            print("Episode finished.")
            break
    
    env.close()

    # Example of how to play interactively (requires a display)
    print("\nTo play interactively, run this script without the 'dummy' video driver.")
    play_interactive = False
    if play_interactive:
        import os
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]
        
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        terminated = False
        total_reward = 0
        
        # Create a window to display the game
        pygame.display.set_caption("Mine Cart Madness")
        display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        
        running = True
        while running:
            # --- Player Input ---
            keys = pygame.key.get_pressed()
            move_action = 0 # none
            if keys[pygame.K_UP]:
                move_action = 1
            elif keys[pygame.K_DOWN]:
                move_action = 2
                
            action = [move_action, 0, 0] # Space and Shift not used

            # --- Game Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # --- Pygame Event Loop ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r and terminated:
                    obs, info = env.reset()
                    terminated = False
                    total_reward = 0

            # --- Render to Display ---
            # The observation is (H, W, C), but pygame blit needs (W, H) surface
            # We can just get the internal surface from the env
            display_screen.blit(env.screen, (0, 0))
            pygame.display.flip()

            if terminated:
                print(f"Game Over! Final Score: {total_reward:.2f}. Press 'R' to restart.")
        
        env.close()