
# Generated: 2025-08-27T23:42:48.797089
# Source Brief: brief_03554.md
# Brief Index: 3554

        
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
        "Controls: ↑ or Space to jump, ↓ or Shift to slide. Survive and reach the end!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, side-scrolling arcade runner. Guide your robot through "
        "obstacle-laden stages, timing your jumps and slides to perfection to "
        "achieve the highest score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    GROUND_Y = 340
    GRAVITY = 0.8
    JUMP_STRENGTH = -15
    
    ROBOT_X_POS = 100
    ROBOT_RUN_SIZE = (30, 50)
    ROBOT_SLIDE_SIZE = (50, 25)
    SLIDE_DURATION = 15  # 0.5 seconds at 30 FPS

    STAGE_COUNT = 3
    STAGE_LENGTH = 8000  # pixels
    MAX_STEPS = 5400 # 3 stages * 60s/stage * 30fps

    OBSTACLE_MIN_GAP = 350
    OBSTACLE_MAX_GAP = 700
    NEAR_MISS_THRESHOLD = 8

    # --- Colors ---
    COLOR_BG = (15, 25, 50)
    COLOR_GROUND = (60, 60, 80)
    COLOR_ROBOT = (0, 150, 255)
    COLOR_ROBOT_GLOW = (100, 200, 255)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (255, 120, 120)
    COLOR_FINISH_LINE = (255, 215, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (20, 20, 30)
    COLOR_PARTICLE_JUMP = (180, 180, 160)
    COLOR_PARTICLE_SLIDE = (255, 200, 100)

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        self.np_random = None
        self.robot_state = None
        self.robot_pos = None
        self.robot_vel = None
        self.slide_timer = None
        self.obstacles = None
        self.particles = None
        self.parallax_bg = None
        self.current_stage = None
        self.stage_progress = None
        self.scroll_speed = None
        self.next_obstacle_spawn_dist = None
        self.next_obstacle_id = None
        self.cleared_obstacles = None
        self.stage_clear_timer = None
        
        self.steps = 0
        self.score = 0
        self.terminated = False

        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.terminated = False
        
        self.current_stage = 1
        self.stage_progress = 0
        self.scroll_speed = 5.0
        
        self.robot_state = "run"
        self.robot_pos = pygame.Vector2(self.ROBOT_X_POS, self.GROUND_Y)
        self.robot_vel = pygame.Vector2(0, 0)
        self.slide_timer = 0
        
        self.obstacles = []
        self.particles = []
        self.next_obstacle_spawn_dist = self.OBSTACLE_MIN_GAP
        self.next_obstacle_id = 0
        self.cleared_obstacles = set()
        
        self.stage_clear_timer = 0
        
        self._init_parallax_bg()
        self._spawn_initial_obstacles()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        self.terminated = False

        self._handle_input(action)
        self._update_game_state()
        
        reward += self._calculate_reward()
        self.score += reward
        
        self.terminated = self._check_termination()
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        is_on_ground = self.robot_pos.y >= self.GROUND_Y

        # Jump action
        if (movement == 1 or space_held) and is_on_ground and self.robot_state != "slide":
            self.robot_state = "jump"
            self.robot_vel.y = self.JUMP_STRENGTH
            # sfx: jump_sound
            self._create_particles(15, self.robot_pos.x + self.ROBOT_RUN_SIZE[0]/2, self.GROUND_Y, self.COLOR_PARTICLE_JUMP)

        # Slide action
        elif (movement == 2 or shift_held) and is_on_ground:
            if self.robot_state != "slide":
                self.robot_state = "slide"
                self.slide_timer = self.SLIDE_DURATION
                # sfx: slide_start_sound
    
    def _update_game_state(self):
        # Update stage clear message
        if self.stage_clear_timer > 0:
            self.stage_clear_timer -= 1
            # Paused state during message
            return
        
        # Update timers and states
        if self.robot_state == "slide":
            self.slide_timer -= 1
            if self.slide_timer <= 0:
                self.robot_state = "run"
                # sfx: slide_end_sound

        # Update robot physics
        self.robot_vel.y += self.GRAVITY
        self.robot_pos.y += self.robot_vel.y
        
        if self.robot_pos.y >= self.GROUND_Y:
            self.robot_pos.y = self.GROUND_Y
            self.robot_vel.y = 0
            if self.robot_state == "jump":
                self.robot_state = "run"
                # sfx: land_sound
                self._create_particles(10, self.robot_pos.x + self.ROBOT_RUN_SIZE[0]/2, self.GROUND_Y, self.COLOR_PARTICLE_JUMP)

        # Update world scroll
        self.stage_progress += self.scroll_speed
        
        # Update obstacles
        for obs in self.obstacles:
            obs['rect'].x -= self.scroll_speed
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > 0]
        
        # Spawn new obstacles
        if self.stage_progress > self.next_obstacle_spawn_dist and self.stage_progress < self.STAGE_LENGTH - self.WIDTH:
            self._spawn_obstacle()
            self.next_obstacle_spawn_dist += self.np_random.integers(self.OBSTACLE_MIN_GAP, self.OBSTACLE_MAX_GAP)

        # Update particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        # Update parallax background
        for layer in self.parallax_bg:
            for item in layer['items']:
                item.x -= self.scroll_speed * layer['speed']
                if item.right < 0:
                    item.x += self.WIDTH + item.width + 50

    def _calculate_reward(self):
        if self.stage_clear_timer > 0:
            return 0.0

        reward = 0.1  # Survival reward

        robot_rect = self._get_robot_rect()

        # Check for cleared obstacles and near misses
        for obs in self.obstacles:
            if obs['id'] not in self.cleared_obstacles and obs['rect'].right < robot_rect.left:
                reward += 1.0  # Cleared obstacle
                self.cleared_obstacles.add(obs['id'])
                # sfx: obstacle_clear_ding

                # Check for near miss
                near_miss = False
                if obs['type'] == 'low': # check top
                    if abs(robot_rect.bottom - obs['rect'].top) < self.NEAR_MISS_THRESHOLD:
                        near_miss = True
                elif obs['type'] == 'high': # check bottom
                    if abs(obs['rect'].bottom - robot_rect.top) < self.NEAR_MISS_THRESHOLD:
                        near_miss = True
                
                if near_miss:
                    reward -= 5.0 # Risky near-miss penalty
                    # sfx: near_miss_whoosh

        # Check for collisions
        if self._check_collisions(robot_rect):
            self.terminated = True
            # sfx: robot_crash_explosion
            return 0 # No specific death penalty, just termination.
        
        # Check for stage completion
        if self.stage_progress >= self.STAGE_LENGTH:
            self.current_stage += 1
            self.stage_progress = 0
            self.obstacles = []
            self.cleared_obstacles = set()
            self.next_obstacle_spawn_dist = self.OBSTACLE_MIN_GAP
            
            if self.current_stage > self.STAGE_COUNT:
                # Game won
                self.terminated = True
                # sfx: game_win_fanfare
                return 100.0
            else:
                # Stage cleared
                reward += 10.0
                self.stage_clear_timer = self.FPS * 2 # 2 second pause
                self.scroll_speed += 0.5 # Increase difficulty
                # sfx: stage_clear_jingle
        
        return reward

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            self.terminated = True
        return self.terminated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._render_parallax_bg()
        self._render_ground()
        self._render_particles()
        if self.stage_progress + self.WIDTH > self.STAGE_LENGTH:
            self._render_finish_line()
        self._render_obstacles()
        if not (self.terminated and not (self.current_stage > self.STAGE_COUNT)):
            self._render_robot()
    
    def _render_ui(self):
        # Stage Text
        self._draw_text(f"Stage: {self.current_stage}/{self.STAGE_COUNT}", (10, 10), self.font_small)
        
        # Score Text
        score_str = f"Score: {int(self.score)}"
        self._draw_text(score_str, (self.WIDTH / 2, 10), self.font_small, center_x=True)

        # Timer Text
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_str = f"Time: {time_left:.1f}"
        self._draw_text(time_str, (self.WIDTH - 10, 10), self.font_small, right_align=True)

        # Stage Clear Message
        if self.stage_clear_timer > self.FPS * 0.25: # Fade in/out effect
            alpha = 255
            if self.stage_clear_timer > self.FPS * 1.75:
                alpha = int(255 * ( (self.FPS*2 - self.stage_clear_timer) / (self.FPS*0.25) ))
            elif self.stage_clear_timer < self.FPS * 0.5:
                 alpha = int(255 * ( (self.stage_clear_timer - self.FPS*0.25) / (self.FPS*0.25) ))

            msg = "STAGE CLEAR!"
            if self.current_stage > self.STAGE_COUNT:
                msg = "YOU WIN!"
            
            text_surf = self.font_large.render(msg, True, self.COLOR_FINISH_LINE)
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stage": self.current_stage}

    # --- Helper Methods ---

    def _get_robot_rect(self):
        if self.robot_state == "slide":
            w, h = self.ROBOT_SLIDE_SIZE
            return pygame.Rect(self.robot_pos.x, self.robot_pos.y - h, w, h)
        else:
            w, h = self.ROBOT_RUN_SIZE
            return pygame.Rect(self.robot_pos.x, self.robot_pos.y - h, w, h)

    def _spawn_obstacle(self):
        obs_type = "low" if self.np_random.random() > 0.5 else "high"
        if obs_type == "low":
            h = self.np_random.integers(30, 50)
            w = self.np_random.integers(40, 60)
            rect = pygame.Rect(self.WIDTH, self.GROUND_Y - h, w, h)
        else: # high
            h = self.np_random.integers(50, 80)
            w = self.np_random.integers(40, 70)
            rect = pygame.Rect(self.WIDTH, self.GROUND_Y - self.ROBOT_RUN_SIZE[1] - 30, w, h)
        
        self.obstacles.append({'rect': rect, 'type': obs_type, 'id': self.next_obstacle_id})
        self.next_obstacle_id += 1

    def _spawn_initial_obstacles(self):
        while self.stage_progress < self.WIDTH * 1.5:
            self.stage_progress += self.np_random.integers(self.OBSTACLE_MIN_GAP, self.OBSTACLE_MAX_GAP)
            self._spawn_obstacle()
            self.obstacles[-1]['rect'].x = self.stage_progress
        self.stage_progress = 0

    def _check_collisions(self, robot_rect):
        for obs in self.obstacles:
            if robot_rect.colliderect(obs['rect']):
                return True
        return False

    def _create_particles(self, count, x, y, color):
        for _ in range(count):
            self.particles.append({
                'pos': pygame.Vector2(x, y),
                'vel': pygame.Vector2(self.np_random.uniform(-2.5, 2.5), self.np_random.uniform(-3, 0)),
                'life': self.np_random.integers(10, 20),
                'color': color
            })

    def _init_parallax_bg(self):
        self.parallax_bg = [
            {'speed': 0.2, 'color': (30, 40, 70), 'items': []},
            {'speed': 0.4, 'color': (45, 55, 90), 'items': []},
        ]
        for layer in self.parallax_bg:
            for _ in range(10):
                w = self.np_random.integers(50, 150)
                h = self.np_random.integers(20, 100)
                x = self.np_random.integers(0, self.WIDTH + w)
                y = self.np_random.integers(self.GROUND_Y - h - 150, self.GROUND_Y - h)
                layer['items'].append(pygame.Rect(x, y, w, h))

    # --- Rendering Helpers ---

    def _render_parallax_bg(self):
        for layer in self.parallax_bg:
            for item in layer['items']:
                pygame.draw.rect(self.screen, layer['color'], item)

    def _render_ground(self):
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

    def _render_robot(self):
        rect = self._get_robot_rect()
        
        # Simple animation for running
        if self.robot_state == "run" and self.robot_pos.y >= self.GROUND_Y:
            bob = math.sin(self.steps * 0.5) * 2
            rect.y -= bob

        # Glow effect
        glow_radius = int(max(rect.width, rect.height) * 0.75)
        pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, glow_radius, (*self.COLOR_ROBOT_GLOW, 50))
        pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, int(glow_radius * 0.7), (*self.COLOR_ROBOT_GLOW, 70))
        
        # Main body
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, rect, border_radius=5)
    
    def _render_obstacles(self):
        for obs in self.obstacles:
            # Glow
            glow_rect = obs['rect'].inflate(8, 8)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_OBSTACLE_GLOW, 80), s.get_rect(), border_radius=10)
            self.screen.blit(s, glow_rect.topleft)
            # Body
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'], border_radius=3)

    def _render_finish_line(self):
        x = self.STAGE_LENGTH - self.stage_progress
        if x < self.WIDTH:
            for i in range(0, self.HEIGHT, 20):
                color = self.COLOR_FINISH_LINE if (i // 20) % 2 == 0 else self.COLOR_BG
                pygame.draw.rect(self.screen, color, (x, i, 10, 20))

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 20))
            size = max(1, int(p['life'] / 4))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), size, (*p['color'], alpha))

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center_x=False, right_align=False):
        shadow_surf = font.render(text, True, shadow_color)
        text_surf = font.render(text, True, color)
        
        if center_x:
            text_rect = text_surf.get_rect(centerx=pos[0], top=pos[1])
        elif right_align:
            text_rect = text_surf.get_rect(right=pos[0], top=pos[1])
        else:
            text_rect = text_surf.get_rect(topleft=pos)

        shadow_rect = text_rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)
        
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Robot Runner")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    # --- Main Game Loop ---
    while not terminated:
        movement, space, shift = 0, 0, 0
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        if keys[pygame.K_r]: # Reset button
            obs, info = env.reset()
            total_reward = 0
            continue

        action = [movement, space, shift]
        
        # --- Step Environment ---
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        terminated = term
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate ---
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {int(total_reward)}")
    pygame.quit()