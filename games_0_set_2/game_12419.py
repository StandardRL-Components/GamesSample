import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:55:35.041810
# Source Brief: brief_02419.md
# Brief Index: 2419
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for a conveyor belt sorting game.

    The player's goal is to sort colored cubes onto their matching targets
    by activating the correct conveyor belts. Successful chains of sorting
    increase a score multiplier. The game ends when a target score is
    reached or when the 30-second timer runs out.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Select conveyor (1: Left/Fast, 2: Mid/Medium, 3: Right/Slow)
    - action[1]: Activate selected conveyor (1: Active, 0: Inactive)
    - action[2]: Unused

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Sort colored cubes onto their matching targets by activating the correct conveyor belts. "
        "Score points and build a combo multiplier before the timer runs out."
    )
    user_guide = (
        "Controls: Use ↑, ↓, and ← arrow keys to select the left, middle, and right conveyor belts. "
        "Hold space to activate the selected conveyor and move the cubes."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    TARGET_SCORE = 5000
    TIME_LIMIT_SECONDS = 30
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_MULTIPLIER = (255, 200, 0)
    COLOR_TIMER_WARN = (255, 100, 100)
    COLOR_CONVEYOR = (60, 70, 80)
    COLOR_CONVEYOR_SELECTED = (255, 255, 0)
    COLOR_CONVEYOR_ACTIVE_DASH = (150, 160, 170)
    
    CUBE_COLORS = {
        "red": (255, 70, 70),
        "green": (70, 255, 70),
        "blue": (70, 70, 255)
    }
    TARGET_COLORS = {
        "red": (180, 40, 40),
        "green": (40, 180, 40),
        "blue": (40, 40, 180)
    }

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
        self.font_large = pygame.font.Font(None, 56)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        self.cubes = []
        self.particles = []
        self.conveyors = []
        self.selected_conveyor_idx = 0
        self.multiplier = 1
        self.last_match_time = 0
        self.cube_spawn_rate = 0
        self.cube_spawn_timer = 0
        self.step_reward = 0
        self.dash_offset = 0

        self._setup_conveyors()
        
        # self.reset() # reset is called by the wrapper/runner
        # self.validate_implementation() # validation is done by the test suite

    def _setup_conveyors(self):
        conveyor_width = 80
        conveyor_height = 300
        spacing = (self.WIDTH - 3 * conveyor_width) / 4
        y_pos = 50

        speeds = [6, 4, 2] # Fast, Medium, Slow (pixels/step)
        colors = ["red", "green", "blue"]

        for i in range(3):
            x_pos = spacing * (i + 1) + conveyor_width * i
            conveyor_rect = pygame.Rect(x_pos, y_pos, conveyor_width, conveyor_height)
            target_rect = pygame.Rect(x_pos, y_pos + conveyor_height, conveyor_width, 50)
            self.conveyors.append({
                "rect": conveyor_rect,
                "target_rect": target_rect,
                "speed": speeds[i],
                "target_color_name": colors[i],
                "target_color_rgb": self.TARGET_COLORS[colors[i]]
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.TIME_LIMIT_SECONDS
        self.cubes = []
        self.particles = []
        self.selected_conveyor_idx = 0
        self.multiplier = 1
        self.last_match_time = -999
        self.cube_spawn_rate = 0.5  # Initial rate
        self.cube_spawn_timer = 1.0 / self.cube_spawn_rate

        # Spawn a few initial cubes
        for _ in range(3):
            self._spawn_cube()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.step_reward = 0
        self.steps += 1
        
        # --- 1. Handle Input ---
        movement, space_held, _ = action
        is_conveyor_active = (space_held == 1)

        if 1 <= movement <= 3:
            self.selected_conveyor_idx = movement - 1

        # --- 2. Update Game State ---
        self.timer -= 1.0 / self.FPS
        self.dash_offset = (self.dash_offset + 2) % 40

        # Difficulty scaling: spawn rate increases over time
        self.cube_spawn_rate = 0.5 + (self.steps / self.FPS) * 0.01
        self.cube_spawn_timer -= 1.0 / self.FPS
        if self.cube_spawn_timer <= 0:
            self._spawn_cube()
            self.cube_spawn_timer = 1.0 / self.cube_spawn_rate

        # --- 3. Update Cubes ---
        cubes_to_remove = []
        for i, cube in enumerate(self.cubes):
            if is_conveyor_active and cube['conveyor_idx'] == self.selected_conveyor_idx:
                speed = self.conveyors[self.selected_conveyor_idx]['speed']
                cube['pos'].y += speed
                
                # Continuous reward for moving a correct cube
                if cube['color_name'] == self.conveyors[self.selected_conveyor_idx]['target_color_name']:
                    self.step_reward += 0.01

            # Check for target collision
            target_rect = self.conveyors[cube['conveyor_idx']]['target_rect']
            if cube['pos'].y + cube['size'] > target_rect.top:
                self._handle_cube_target_collision(cube)
                cubes_to_remove.append(cube)

        self.cubes = [c for c in self.cubes if c not in cubes_to_remove]
        
        # --- 4. Update Particles ---
        self._update_particles()

        # --- 5. Check Termination & Calculate Final Reward ---
        terminated = False
        reward = self.step_reward
        
        if self.score >= self.TARGET_SCORE:
            self.game_over = True
            terminated = True
            reward += 100 # Goal-oriented reward
            # SFX: Win Jingle
        elif self.timer <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            reward = -10 # Timeout penalty
            # SFX: Lose Buzzer
        
        # Clamp non-terminal rewards
        if not terminated:
            reward = max(-10.0, min(10.0, reward))

        truncated = self.steps >= self.MAX_STEPS
        if terminated:
            truncated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_cube(self):
        conveyor_idx = self.np_random.integers(0, 3)
        color_name = self.np_random.choice(list(self.CUBE_COLORS.keys()))
        color_rgb = self.CUBE_COLORS[color_name]
        
        conveyor_rect = self.conveyors[conveyor_idx]['rect']
        size = 30
        
        # Ensure no overlap at spawn
        x_pos = conveyor_rect.centerx
        y_pos = conveyor_rect.top - size
        for other_cube in self.cubes:
            if other_cube['conveyor_idx'] == conveyor_idx and other_cube['pos'].y < conveyor_rect.top:
                return # Don't spawn if space is occupied

        self.cubes.append({
            'pos': pygame.Vector2(x_pos, y_pos),
            'color_name': color_name,
            'color_rgb': color_rgb,
            'size': size,
            'conveyor_idx': conveyor_idx
        })

    def _handle_cube_target_collision(self, cube):
        target_info = self.conveyors[cube['conveyor_idx']]
        is_match = (cube['color_name'] == target_info['target_color_name'])
        
        if is_match:
            # SFX: Success Chime
            time_since_last_match = self.steps - self.last_match_time
            if time_since_last_match < self.FPS * 2: # 2-second chain window
                self.multiplier += 1
            else:
                self.multiplier = 1
            
            self.last_match_time = self.steps
            
            match_score = 10 * self.multiplier
            self.score += match_score
            self.step_reward += self.multiplier # Per brief: "+1 multiplied by the current multiplier"
            
            self._create_particles(cube['pos'], cube['color_rgb'], 30, 5)
        else:
            # SFX: Mismatch Buzz
            self.multiplier = 1
            self.step_reward -= 1.0 # Penalty for mismatch
            self._create_particles(cube['pos'], (100, 100, 100), 15, 2)

    def _create_particles(self, pos, color, count, speed_factor):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_factor)
            velocity = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': velocity,
                'color': color,
                'lifespan': self.np_random.uniform(20, 40),
                'radius': self.np_random.uniform(3, 7)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['lifespan'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['radius'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw conveyors and targets
        for i, c in enumerate(self.conveyors):
            pygame.draw.rect(self.screen, c['target_color_rgb'], c['target_rect'])
            pygame.draw.rect(self.screen, self.COLOR_CONVEYOR, c['rect'])
            
            # Blinking effect for selection
            if self.selected_conveyor_idx == i and (self.steps % 10 < 5):
                 pygame.gfxdraw.rectangle(self.screen, c['rect'], self.COLOR_CONVEYOR_SELECTED)
            
            # Draw animated dashes for visual movement feedback
            for y_offset in range(0, c['rect'].height, 40):
                y = c['rect'].top + (y_offset + self.dash_offset) % c['rect'].height
                start_pos = (c['rect'].centerx, y)
                end_pos = (c['rect'].centerx, y + 20)
                if c['rect'].collidepoint(start_pos) and c['rect'].collidepoint(end_pos):
                    pygame.draw.line(self.screen, self.COLOR_CONVEYOR_ACTIVE_DASH, start_pos, end_pos, 3)

        # Draw cubes
        for cube in self.cubes:
            rect = pygame.Rect(0, 0, cube['size'], cube['size'])
            rect.center = (int(cube['pos'].x), int(cube['pos'].y))
            
            # Simple 3D effect
            darker_color = tuple(max(0, val - 50) for val in cube['color_rgb'])
            pygame.draw.rect(self.screen, darker_color, rect.move(3, 3))
            pygame.draw.rect(self.screen, cube['color_rgb'], rect)
            pygame.draw.rect(self.screen, (255,255,255), rect, 1)

        # Draw particles
        for p in self.particles:
            # Using gfxdraw for anti-aliased circles with alpha
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 40.0))))
            color_with_alpha = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color_with_alpha)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        timer_color = self.COLOR_UI_TEXT if self.timer > 10 else self.COLOR_TIMER_WARN
        timer_text = self.font_medium.render(f"Time: {max(0, self.timer):.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Multiplier
        if self.multiplier > 1:
            multi_text = self.font_large.render(f"x{self.multiplier}", True, self.COLOR_MULTIPLIER)
            
            # Pop effect for feedback
            time_since_match = self.steps - self.last_match_time
            if time_since_match < self.FPS / 2: # Half-second pop
                scale = 1.0 + 0.3 * math.sin(time_since_match / (self.FPS / 2) * math.pi)
                scaled_text = pygame.transform.rotozoom(multi_text, 0, scale)
                text_rect = scaled_text.get_rect(center=(self.WIDTH // 2, 30))
                self.screen.blit(scaled_text, text_rect)
            else:
                text_rect = multi_text.get_rect(center=(self.WIDTH // 2, 30))
                self.screen.blit(multi_text, text_rect)
        
        # Game Over/Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "LEVEL COMPLETE!" if self.score >= self.TARGET_SCORE else "TIME'S UP!"
            end_text = self.font_large.render(message, True, (255, 255, 255))
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "multiplier": self.multiplier
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2], f"Action space nvec is {self.action_space.nvec.tolist()}"
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8, f"Obs dtype is {test_obs.dtype}"
        
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # For manual play, we need a real display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Conveyor Sort")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # --- Manual Control Mapping ---
    selected_conveyor_action = 0 # 0=None, 1=Left, 2=Mid, 3=Right
    space_held = 0

    print("--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("Q: Quit")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Environment Reset ---")
                if event.key == pygame.K_UP:
                    selected_conveyor_action = 1
                if event.key == pygame.K_DOWN:
                    selected_conveyor_action = 2
                if event.key == pygame.K_LEFT:
                    selected_conveyor_action = 3
                if event.key == pygame.K_SPACE:
                    space_held = 1
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT]:
                    selected_conveyor_action = 0
                if event.key == pygame.K_SPACE:
                    space_held = 0

        action = [selected_conveyor_action, space_held, 0]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode Finished! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)
        
    env.close()