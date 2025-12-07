
# Generated: 2025-08-27T21:18:25.398758
# Source Brief: brief_02744.md
# Brief Index: 2744

        
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
        "Controls: ←→ to run. Press space to jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, side-scrolling platformer. Guide the robot to the finish line, jumping over obstacles. Complete 3 stages to win."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_Y = 350
    
    # Colors
    COLOR_BG_SKY = (135, 206, 235)
    COLOR_GROUND = (101, 67, 33)
    COLOR_MOUNTAIN_1 = (100, 100, 120)
    COLOR_MOUNTAIN_2 = (120, 120, 140)
    
    COLOR_ROBOT_BODY = (0, 120, 255)
    COLOR_ROBOT_LIMB = (50, 150, 255)
    COLOR_ROBOT_EYE = (255, 255, 255)

    OBSTACLE_COLORS = {
        "low": (220, 50, 50),
        "high": (240, 180, 50),
        "flying": (50, 200, 100)
    }
    
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (50, 50, 50)
    COLOR_FINISH_LINE = (255, 255, 255)

    # Physics
    GRAVITY = 0.6
    JUMP_STRENGTH = -12
    ROBOT_SPEED = 4.0
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24)
            self.font_stage = pygame.font.SysFont("Consolas", 32, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 28)
            self.font_stage = pygame.font.SysFont(None, 36)

        self._generate_background_assets()
        self.reset()
        
        self.validate_implementation()

    def _generate_background_assets(self):
        self.mountains = []
        for _ in range(10):
            x = random.randint(-self.SCREEN_WIDTH, self.SCREEN_WIDTH * 3)
            y = self.GROUND_Y
            width = random.randint(200, 400)
            height = random.randint(100, 250)
            color = random.choice([self.COLOR_MOUNTAIN_1, self.COLOR_MOUNTAIN_2])
            self.mountains.append((pygame.Rect(x, y - height, width, height), color))
            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        
        self.stage = 1
        self.world_scroll = 0.0
        self.finish_line_pos = 3000 * self.stage

        self.robot_pos = [self.SCREEN_WIDTH // 2, self.GROUND_Y]
        self.robot_vel = [0, 0]
        self.on_ground = True
        
        self.obstacles = []
        self.particles = []
        
        self.obstacle_spawn_timer = 60
        self.difficulty_speed_bonus = 0.0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self.reward_this_step = 0
        terminated = False

        # --- Update Game Logic ---
        self._update_difficulty()
        self._handle_input_and_update_robot(movement, space_held)
        self._update_particles()
        self._update_obstacles()
        
        # --- Check Game State and Calculate Rewards ---
        collision = self._check_collisions()
        if collision:
            self.reward_this_step -= 5.0 # Event-based reward for collision
            # sfx: robot_crash
            terminated = True
        
        stage_completed = self._check_stage_completion()
        if stage_completed:
            # sfx: stage_clear
            if self.stage < 3:
                self.reward_this_step += 50.0 # Goal-oriented reward
                self._advance_stage()
            else:
                self.reward_this_step += 100.0 # Final goal reward
                terminated = True

        if self.steps >= 2000:
            terminated = True

        self.score += self.reward_this_step
        self.steps += 1
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 500 == 0:
            self.difficulty_speed_bonus += 0.05

    def _handle_input_and_update_robot(self, movement, space_held):
        robot_dx = 0
        if movement == 3: # Left
            robot_dx = -self.ROBOT_SPEED
            self.reward_this_step += 0.0 # No reward/penalty for moving left
        elif movement == 4: # Right
            robot_dx = self.ROBOT_SPEED
            self.reward_this_step += 0.1 # Continuous reward for moving forward
        else: # Still
            self.reward_this_step -= 0.02 # Continuous penalty for standing still

        self.world_scroll += robot_dx
        self.world_scroll = max(0, self.world_scroll)

        # Jumping
        if space_held and self.on_ground:
            self.robot_vel[1] = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump
            self._spawn_particles(self.robot_pos[0], self.robot_pos[1] + 20, 5, self.COLOR_ROBOT_LIMB, 'jump')

        # Physics
        self.robot_vel[1] += self.GRAVITY
        self.robot_pos[1] += self.robot_vel[1]

        # Ground collision
        if self.robot_pos[1] >= self.GROUND_Y:
            if not self.on_ground:
                # sfx: land
                self._spawn_particles(self.robot_pos[0], self.GROUND_Y, 8, self.COLOR_GROUND, 'land')
            self.robot_pos[1] = self.GROUND_Y
            self.robot_vel[1] = 0
            self.on_ground = True

    def _update_obstacles(self):
        # Spawn new obstacles
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            self._spawn_obstacle()
            spawn_time_reduction = (self.stage - 1) * 10 + (self.difficulty_speed_bonus * 50)
            self.obstacle_spawn_timer = random.randint(60, 120) - spawn_time_reduction

        # Update existing obstacles
        for obs in self.obstacles:
            if not obs['cleared'] and (self.robot_pos[0] > obs['rect'].right - self.world_scroll):
                obs['cleared'] = True
                self.reward_this_step += 1.0 # Event-based reward for clearing obstacle
                # sfx: clear_obstacle
        
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right - self.world_scroll > 0]

    def _spawn_obstacle(self):
        obs_type = random.choice(["low", "low", "high", "flying"])
        x = self.world_scroll + self.SCREEN_WIDTH + 100
        speed = random.uniform(1.5, 2.5) + self.difficulty_speed_bonus + (self.stage - 1) * 0.5
        
        if obs_type == "low":
            w, h = 40, 40
            y = self.GROUND_Y - h
        elif obs_type == "high":
            w, h = 40, 80
            y = self.GROUND_Y - h
        else: # flying
            w, h = 50, 30
            y = random.choice([self.GROUND_Y - 60, self.GROUND_Y - 120])
        
        self.obstacles.append({
            'rect': pygame.Rect(x, y, w, h),
            'type': obs_type,
            'cleared': False
        })
    
    def _check_collisions(self):
        robot_hitbox = self._get_robot_hitbox()
        for obs in self.obstacles:
            obs_screen_rect = obs['rect'].copy()
            obs_screen_rect.x -= int(self.world_scroll)
            if robot_hitbox.colliderect(obs_screen_rect):
                return True
        return False
        
    def _check_stage_completion(self):
        return self.world_scroll + self.robot_pos[0] > self.finish_line_pos

    def _advance_stage(self):
        self.stage += 1
        self.world_scroll = 0
        self.obstacles.clear()
        self.particles.clear()
        self.finish_line_pos = 3000 * self.stage
        self.obstacle_spawn_timer = 60

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['type'] == 'land':
                p['vel'][1] += 0.1 # particle gravity
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _spawn_particles(self, x, y, count, color, p_type):
        for _ in range(count):
            if p_type == 'land':
                vel = [random.uniform(-2, 2), random.uniform(-2, 0)]
            elif p_type == 'jump':
                vel = [random.uniform(-1, 1), random.uniform(1, 2)]
            else:
                vel = [0,0]
            self.particles.append({
                'pos': [x, y], 'vel': vel, 'life': random.randint(15, 30),
                'color': color, 'size': random.randint(2, 5), 'type': p_type
            })

    def _get_robot_hitbox(self):
        return pygame.Rect(self.robot_pos[0] - 15, self.robot_pos[1] - 40, 30, 40)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG_SKY)
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Parallax mountains
        for rect, color in self.mountains:
            screen_x = rect.x - int(self.world_scroll * 0.2)
            if -rect.width < screen_x < self.SCREEN_WIDTH:
                pygame.draw.polygon(self.screen, color, [
                    (screen_x, rect.bottom),
                    (screen_x + rect.width / 2, rect.top),
                    (screen_x + rect.width, rect.bottom)
                ])
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))

    def _render_game_elements(self):
        # Particles
        for p in self.particles:
            size = max(0, p['size'] * (p['life'] / 30.0))
            pygame.draw.rect(self.screen, p['color'], (int(p['pos'][0] - size/2), int(p['pos'][1] - size/2), int(size), int(size)))

        # Finish line
        finish_screen_x = self.finish_line_pos - self.world_scroll
        if 0 < finish_screen_x < self.SCREEN_WIDTH:
            for i in range(10):
                color = self.COLOR_FINISH_LINE if i % 2 == 0 else (0,0,0)
                pygame.draw.rect(self.screen, color, (finish_screen_x, self.GROUND_Y - 100 + i * 10, 10, 10))
            pygame.draw.line(self.screen, (50,50,50), (finish_screen_x, self.GROUND_Y), (finish_screen_x, self.GROUND_Y-100), 2)

        # Obstacles
        for obs in self.obstacles:
            obs_screen_rect = obs['rect'].copy()
            obs_screen_rect.x -= int(self.world_scroll)
            color = self.OBSTACLE_COLORS[obs['type']]
            pygame.draw.rect(self.screen, color, obs_screen_rect)
            pygame.draw.rect(self.screen, tuple(max(0, c-40) for c in color), obs_screen_rect, 2)
            
        # Robot
        self._render_robot()
    
    def _render_robot(self):
        x, y = int(self.robot_pos[0]), int(self.robot_pos[1])
        
        # Body
        body_rect = pygame.Rect(x - 15, y - 40, 30, 30)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT_BODY, body_rect, border_radius=4)
        
        # Eye
        eye_x = x + 5
        pygame.draw.circle(self.screen, self.COLOR_ROBOT_EYE, (eye_x, y - 25), 5)
        pygame.draw.circle(self.screen, (0,0,0), (eye_x + 1, y - 25), 2)
        
        # Legs
        if self.on_ground:
            leg_angle = math.sin(self.steps * 0.4) * 25
        else:
            leg_angle = 0 # Tucked in while jumping

        leg1_start = (x - 7, y - 10)
        leg1_end = (leg1_start[0] + 15 * math.sin(math.radians(leg_angle)),
                    leg1_start[1] + 15 * math.cos(math.radians(leg_angle)))
        
        leg2_start = (x + 7, y - 10)
        leg2_end = (leg2_start[0] + 15 * math.sin(math.radians(-leg_angle)),
                    leg2_start[1] + 15 * math.cos(math.radians(-leg_angle)))

        pygame.draw.line(self.screen, self.COLOR_ROBOT_LIMB, leg1_start, leg1_end, 6)
        pygame.draw.line(self.screen, self.COLOR_ROBOT_LIMB, leg2_start, leg2_end, 6)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, x, y, anchor="topleft"):
            text_surf = font.render(text, True, self.COLOR_TEXT)
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_rect = text_surf.get_rect()
            
            if anchor == "topright":
                text_rect.topright = (x, y)
            elif anchor == "topleft":
                text_rect.topleft = (x, y)
            
            shadow_rect = text_rect.copy()
            shadow_rect.x += 2
            shadow_rect.y += 2
            
            self.screen.blit(shadow_surf, shadow_rect)
            self.screen.blit(text_surf, text_rect)

        # Stage display
        draw_text(f"STAGE {self.stage}", self.font_stage, 20, 10, "topleft")

        # Score and Steps display
        draw_text(f"SCORE: {int(self.score)}", self.font_ui, self.SCREEN_WIDTH - 20, 10, "topright")
        draw_text(f"STEPS: {self.steps}", self.font_ui, self.SCREEN_WIDTH - 20, 40, "topright")

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "progress": (self.world_scroll + self.robot_pos[0]) / self.finish_line_pos
        }
        
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Robot Runner")
    
    obs, info = env.reset()
    terminated = False
    
    print(env.user_guide)
    
    while not terminated:
        # Map keyboard keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        
        movement_action = 0 # none
        if keys[pygame.K_RIGHT]:
            movement_action = 4
        elif keys[pygame.K_LEFT]:
            movement_action = 3
            
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        env.clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
    pygame.quit()