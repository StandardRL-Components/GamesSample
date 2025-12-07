
# Generated: 2025-08-27T20:32:59.701800
# Source Brief: brief_02494.md
# Brief Index: 2494

        
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
        "Controls: ↑↓ to aim, ←→ to set power. Space to shoot. Shift to reset aim."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hit 10 targets with a limited number of arrows. Earn more points for hitting the bullseye."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_SKY = (135, 206, 235)
    COLOR_BG_GROUND = (34, 139, 34)
    COLOR_PLAYER = (70, 70, 70)
    COLOR_BOWSTRING = (20, 20, 20)
    COLOR_ARROW = (139, 69, 19)
    COLOR_TRAIL = (255, 255, 255, 100)
    
    COLOR_TARGET_OUTER = (255, 255, 255)
    COLOR_TARGET_MIDDLE = (255, 0, 0)
    COLOR_TARGET_INNER = (255, 255, 0) # Bullseye
    COLOR_TARGET_STAND = (160, 82, 45)
    COLOR_TARGET_HIT = (60, 60, 60)
    
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SHADOW = (0, 0, 0)
    
    # Game Parameters
    NUM_TARGETS = 10
    MAX_ARROWS = 15
    MAX_STEPS = 1000
    GRAVITY = 0.1
    
    PLAYER_POS = (60, SCREEN_HEIGHT - 80)
    MIN_ANGLE = -80
    MAX_ANGLE = 5
    MIN_POWER = 5
    MAX_POWER = 25
    
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Etc...        
        
        # Initialize state variables
        self.reset()

        # Run self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state, for example:
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.aim_angle = -30.0
        self.shot_power = (self.MIN_POWER + self.MAX_POWER) / 2
        self.arrows_left = self.MAX_ARROWS
        self.targets_hit_count = 0
        self.arrow = None
        self.particles = []
        self.last_shot_feedback = ""
        
        # Generate new targets
        self.targets = []
        for i in range(self.NUM_TARGETS):
            is_overlapping = True
            # Attempt to place a target without overlapping
            for _ in range(100): # Limit attempts to prevent infinite loop
                x = self.np_random.integers(low=self.SCREEN_WIDTH // 2, high=self.SCREEN_WIDTH - 50)
                y = self.np_random.integers(low=100, high=self.SCREEN_HEIGHT - 100)
                
                is_overlapping = False
                for t in self.targets:
                    dist = math.hypot(x - t['x'], y - t['y'])
                    if dist < 60: # Minimum distance between target centers
                        is_overlapping = True
                        break
                if not is_overlapping:
                    break
            
            self.targets.append({
                "x": x, "y": y, "hit": False,
                "r_outer": 25, "r_middle": 15, "r_inner": 7
            })
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False
        self.steps += 1
        self.last_shot_feedback = "" # Clear feedback
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # --- Handle player input ---
        is_shooting = space_held and self.arrows_left > 0

        # Cannot aim while an arrow is flying (not an issue with auto_advance=False)
        # Movement: Aim angle
        if movement == 1: # Up
            self.aim_angle = max(self.MIN_ANGLE, self.aim_angle - 1.0)
        elif movement == 2: # Down
            self.aim_angle = min(self.MAX_ANGLE, self.aim_angle + 1.0)
        
        # Movement: Power
        if movement == 3: # Left
            self.shot_power = max(self.MIN_POWER, self.shot_power - 0.5)
        elif movement == 4: # Right
            self.shot_power = min(self.MAX_POWER, self.shot_power + 0.5)

        # Shift: Reset aim/power
        if shift_held:
            self.aim_angle = -30.0
            self.shot_power = (self.MIN_POWER + self.MAX_POWER) / 2
        
        # --- Handle shooting ---
        if is_shooting:
            # SFX: Bow release sound
            self.arrows_left -= 1
            angle_rad = math.radians(self.aim_angle)
            
            self.arrow = {
                "x": self.PLAYER_POS[0], "y": self.PLAYER_POS[1],
                "vx": self.shot_power * math.cos(angle_rad),
                "vy": self.shot_power * math.sin(angle_rad),
            }
            
            # Simulate the entire arrow flight within this step
            shot_reward, hit_info = self._simulate_arrow_flight()
            reward += shot_reward
            self.arrow = None # Arrow flight is over

            if hit_info:
                self.score += hit_info['score']
                self.targets_hit_count += 1
                self.last_shot_feedback = hit_info['message']
                # SFX: Target hit sound
                self._create_particles(hit_info['pos'], hit_info['color'])
            else:
                self.last_shot_feedback = "Miss!"
                # SFX: Arrow swoosh and thud
        
        # --- Update game state and check for termination ---
        if self.targets_hit_count >= self.NUM_TARGETS:
            reward += 50
            terminated = True
            self.game_over = True
            self.last_shot_feedback = "All targets hit! Victory!"
        elif self.arrows_left <= 0 and not is_shooting:
            reward -= 50
            terminated = True
            self.game_over = True
            self.last_shot_feedback = "Out of arrows! Game Over."
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.last_shot_feedback = "Time's up! Game Over."
            
        self._update_particles()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _simulate_arrow_flight(self):
        """Simulates arrow flight until collision or off-screen. Returns (reward, hit_info)."""
        for _ in range(500): # Max simulation steps to prevent infinite loops
            self.arrow['x'] += self.arrow['vx']
            self.arrow['y'] += self.arrow['vy']
            self.arrow['vy'] += self.GRAVITY

            for target in self.targets:
                if not target['hit']:
                    dist = math.hypot(self.arrow['x'] - target['x'], self.arrow['y'] - target['y'])
                    if dist < target['r_outer']:
                        target['hit'] = True
                        if dist < target['r_inner']:
                            return 5, {"score": 10, "message": "Bullseye!", "pos": (target['x'], target['y']), "color": self.COLOR_TARGET_INNER}
                        elif dist < target['r_middle']:
                            return 1, {"score": 5, "message": "Good Hit!", "pos": (target['x'], target['y']), "color": self.COLOR_TARGET_MIDDLE}
                        else:
                            return 1, {"score": 2, "message": "Hit!", "pos": (target['x'], target['y']), "color": self.COLOR_TARGET_OUTER}

            if self.arrow['y'] > self.SCREEN_HEIGHT - 60 or self.arrow['x'] > self.SCREEN_WIDTH or self.arrow['x'] < 0:
                return -1, None # Miss
        
        return -1, None # Miss if simulation times out

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG_SKY)
        pygame.draw.rect(self.screen, self.COLOR_BG_GROUND, (0, self.SCREEN_HEIGHT - 60, self.SCREEN_WIDTH, 60))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._render_targets()
        self._render_player_and_aim()
        self._render_particles()

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }
        
    def _render_targets(self):
        for t in sorted(self.targets, key=lambda trg: trg['y']): # Draw from back to front
            stand_top_y = t['y'] + t['r_outer']
            stand_bottom_y = self.SCREEN_HEIGHT - 60
            pygame.draw.line(self.screen, self.COLOR_TARGET_STAND, (t['x'], stand_top_y), (t['x'], stand_bottom_y), 4)

            color_outer = self.COLOR_TARGET_OUTER if not t['hit'] else self.COLOR_TARGET_HIT
            color_middle = self.COLOR_TARGET_MIDDLE if not t['hit'] else self.COLOR_TARGET_HIT
            color_inner = self.COLOR_TARGET_INNER if not t['hit'] else self.COLOR_TARGET_HIT
            
            pygame.gfxdraw.filled_circle(self.screen, int(t['x']), int(t['y']), int(t['r_outer']), color_outer)
            pygame.gfxdraw.filled_circle(self.screen, int(t['x']), int(t['y']), int(t['r_middle']), color_middle)
            pygame.gfxdraw.filled_circle(self.screen, int(t['x']), int(t['y']), int(t['r_inner']), color_inner)
            pygame.gfxdraw.aacircle(self.screen, int(t['x']), int(t['y']), int(t['r_outer']), (0,0,0))
            pygame.gfxdraw.aacircle(self.screen, int(t['x']), int(t['y']), int(t['r_middle']), (0,0,0))
            pygame.gfxdraw.aacircle(self.screen, int(t['x']), int(t['y']), int(t['r_inner']), (0,0,0))

    def _render_player_and_aim(self):
        # Draw player (simple stick figure)
        player_x, player_y = int(self.PLAYER_POS[0]), int(self.PLAYER_POS[1])
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (player_x, player_y-10), 10)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (player_x, player_y), (player_x, player_y+20), 4)
        
        # Draw Bow
        angle_rad = math.radians(self.aim_angle)
        power_ratio = (self.shot_power - self.MIN_POWER) / (self.MAX_POWER - self.MIN_POWER)
        bow_pull = 5 + 15 * power_ratio
        
        bow_tip_x = player_x + 20 * math.cos(angle_rad)
        bow_tip_y = player_y + 20 * math.sin(angle_rad)
        bow_bottom_x = player_x - 20 * math.cos(angle_rad)
        bow_bottom_y = player_y - 20 * math.sin(angle_rad)
        
        string_x = player_x - bow_pull * math.sin(angle_rad)
        string_y = player_y + bow_pull * math.cos(angle_rad)
        
        pygame.draw.line(self.screen, self.COLOR_BOWSTRING, (bow_tip_x, bow_tip_y), (string_x, string_y), 2)
        pygame.draw.line(self.screen, self.COLOR_BOWSTRING, (bow_bottom_x, bow_bottom_y), (string_x, string_y), 2)
        
        # Draw aiming trajectory preview
        vx = self.shot_power * math.cos(angle_rad)
        vy = self.shot_power * math.sin(angle_rad)
        px, py = string_x, string_y
        for i in range(20):
            px += vx * 0.3
            py += vy * 0.3
            vy += self.GRAVITY * 0.3
            if i % 2 == 0:
                pygame.draw.circle(self.screen, self.COLOR_TRAIL, (int(px), int(py)), 1)
                    
    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['size']))
            
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.1)
            
    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': pos[0], 'y': pos[1],
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(15, 30),
                'size': self.np_random.uniform(2, 5),
                'color': color
            })
            
    def _render_ui(self):
        def draw_text(text, font, color, x, y, center=False):
            text_surf = font.render(text, True, self.COLOR_UI_SHADOW)
            text_rect = text_surf.get_rect()
            if center: text_rect.center = (x + 1, y + 1)
            else: text_rect.topleft = (x + 1, y + 1)
            self.screen.blit(text_surf, text_rect)
            
            text_surf = font.render(text, True, color)
            text_rect = text_surf.get_rect()
            if center: text_rect.center = (x, y)
            else: text_rect.topleft = (x, y)
            self.screen.blit(text_surf, text_rect)

        draw_text(f"Score: {self.score}", self.font_small, self.COLOR_UI_TEXT, 10, 10)
        draw_text(f"Arrows: {self.arrows_left}", self.font_small, self.COLOR_UI_TEXT, 10, 35)
        draw_text(f"Targets Hit: {self.targets_hit_count}/{self.NUM_TARGETS}", self.font_small, self.COLOR_UI_TEXT, 10, 60)

        power_ratio = (self.shot_power - self.MIN_POWER) / (self.MAX_POWER - self.MIN_POWER)
        angle_ratio = (self.aim_angle - self.MIN_ANGLE) / (self.MAX_ANGLE - self.MIN_ANGLE)
        
        draw_text("Power", self.font_small, self.COLOR_UI_TEXT, self.SCREEN_WIDTH - 120, 10)
        pygame.draw.rect(self.screen, self.COLOR_UI_SHADOW, (self.SCREEN_WIDTH - 120, 30, 100, 15))
        pygame.draw.rect(self.screen, self.COLOR_TARGET_MIDDLE, (self.SCREEN_WIDTH - 120, 30, 100 * power_ratio, 15))
        
        draw_text("Angle", self.font_small, self.COLOR_UI_TEXT, self.SCREEN_WIDTH - 120, 55)
        pygame.draw.rect(self.screen, self.COLOR_UI_SHADOW, (self.SCREEN_WIDTH - 120, 75, 100, 15))
        pygame.draw.rect(self.screen, self.COLOR_TARGET_INNER, (self.SCREEN_WIDTH - 120, 75, 100 * angle_ratio, 15))

        if self.last_shot_feedback:
            draw_text(self.last_shot_feedback, self.font_large, self.COLOR_UI_TEXT, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, center=True)

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

# Example of how to run the environment for human play
if __name__ == '__main__':
    import os
    # Set the SDL video driver to a dummy driver if you want to run headless.
    # For interactive play, you might need 'x11', 'directfb', 'windows', etc.
    # os.environ['SDL_VIDEODRIVER'] = 'dummy' 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Archery Game")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0.0
    
    # Initialize action array
    action = env.action_space.sample()
    action.fill(0)

    print("\n" + "="*30)
    print("      ARCHERY CHALLENGE")
    print("="*30)
    print(env.game_description)
    print("\n" + env.user_guide)
    print("Press 'R' to reset. Close window to quit.")
    print("="*30 + "\n")

    while running:
        # --- Human Input ---
        # Since auto_advance is False, we only step when an action is taken.
        # For a responsive human experience, we poll keys and step on each frame.
        action.fill(0) # Reset actions for this frame
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        
        if keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0.0
                print("--- Game Reset ---")

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Episode Finished! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            print("Press 'R' to play again or close the window.")
            wait_for_reset = True
            while wait_for_reset and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0.0
                        print("--- Game Reset ---")
                        wait_for_reset = False
                env.clock.tick(10)

        env.clock.tick(60)

    env.close()