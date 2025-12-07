
# Generated: 2025-08-27T13:07:38.424199
# Source Brief: brief_00270.md
# Brief Index: 270

        
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
        "Controls: ↑↓←→ to hop. Hold Space for a long-distance hyper-jump. Press Shift to activate a temporary shield."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a hopping spaceship through a procedurally generated asteroid field to reach the goal before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    LOGIC_TICK_RATE = 10 # Game logic updates per second

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (200, 255, 255)
    COLOR_PLAYER_SHIELD = (100, 150, 255, 128)
    COLOR_OBSTACLE = (255, 80, 80)
    COLOR_OBSTACLE_OUTLINE = (255, 150, 150)
    COLOR_GOAL = (80, 255, 80)
    COLOR_CHECKPOINT = (80, 200, 255)
    COLOR_SAFE_ZONE = (50, 50, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_PARTICLE_THRUST = (255, 200, 100)
    COLOR_PARTICLE_JUMP = (150, 200, 255)
    COLOR_PARTICLE_EXPLOSION = (255, 100, 50)

    # Game Parameters
    PLAYER_RADIUS = 12
    PLAYER_HOP_DISTANCE = 40
    PLAYER_HYPER_JUMP_MULTIPLIER = 2.5
    HOP_COOLDOWN_TICKS = 3 # 0.3s
    HYPER_JUMP_COOLDOWN_TICKS = 6 # 0.6s
    SHIELD_DURATION_TICKS = 5 * LOGIC_TICK_RATE # 5 seconds
    SHIELD_COST_TICKS = 10 * LOGIC_TICK_RATE # 10 seconds
    HYPER_JUMP_COST_TICKS = 2 * LOGIC_TICK_RATE # 2 seconds

    TOTAL_TIME_SECONDS = 180
    MAX_LOGIC_TICKS = TOTAL_TIME_SECONDS * LOGIC_TICK_RATE

    STAGE_DURATION_SECONDS = 60
    STAGE_DURATION_TICKS = STAGE_DURATION_SECONDS * LOGIC_TICK_RATE

    SAFE_ZONE_MARGIN = 40

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_target_pos = pygame.Vector2(0, 0)
        self.obstacles = []
        self.particles = []
        self.stars = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.logic_ticks = 0
        self.score = 0
        self.game_over = False
        
        self.time_remaining_ticks = self.MAX_LOGIC_TICKS
        
        self.stage = 1
        self.stage_timer = 0
        
        self.player_pos.update(self.SCREEN_WIDTH * 0.1, self.SCREEN_HEIGHT / 2)
        self.player_target_pos.update(self.player_pos)
        
        self.hop_cooldown = 0
        self.shield_active = False
        self.shield_timer = 0
        self.space_was_held = False
        self.shift_was_held = False

        self.goal_rect = pygame.Rect(self.SCREEN_WIDTH - 20, 0, 20, self.SCREEN_HEIGHT)
        self.checkpoint_rect = pygame.Rect(self.SCREEN_WIDTH / 2 - 10, 0, 20, self.SCREEN_HEIGHT)
        self.checkpoint_reached = False
        
        self.last_dist_to_goal = self.player_pos.distance_to(pygame.Vector2(self.goal_rect.center))

        self.obstacles = []
        self.particles = []
        self._setup_stage()
        self._setup_starfield()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.game_over = self.time_remaining_ticks <= 0

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_pressed = action[1] == 1 and not self.space_was_held
            shift_pressed = action[2] == 1 and not self.shift_was_held
            self.space_was_held = action[1] == 1
            self.shift_was_held = action[2] == 1

            self._handle_input(movement, space_pressed, shift_pressed)
            self._update_game_state()
            reward = self._calculate_reward()
            self._check_collisions()
            self._update_stage()

        self.steps += 1
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, movement, space_pressed, shift_pressed):
        if self.hop_cooldown > 0:
            return

        hop_direction = pygame.Vector2(0, 0)
        hop_executed = False

        if space_pressed and self.time_remaining_ticks > self.HYPER_JUMP_COST_TICKS:
            # Hyper-jump uses last movement direction
            last_move_dir = pygame.Vector2(0, 0)
            if movement == 1: last_move_dir.y = -1
            elif movement == 2: last_move_dir.y = 1
            elif movement == 3: last_move_dir.x = -1
            elif movement == 4: last_move_dir.x = 1
            
            if last_move_dir.length() > 0:
                hop_direction = last_move_dir
                self.player_target_pos += hop_direction * self.PLAYER_HOP_DISTANCE * self.PLAYER_HYPER_JUMP_MULTIPLIER
                self.time_remaining_ticks -= self.HYPER_JUMP_COST_TICKS
                self.hop_cooldown = self.HYPER_JUMP_COOLDOWN_TICKS
                self._create_particles(self.player_pos, 30, self.COLOR_PARTICLE_JUMP, (15, 30), (2, 5), hop_direction)
                # SFX: Hyper Jump
                hop_executed = True

        elif movement != 0:
            if movement == 1: hop_direction.y = -1
            elif movement == 2: hop_direction.y = 1
            elif movement == 3: hop_direction.x = -1
            elif movement == 4: hop_direction.x = 1
            
            self.player_target_pos += hop_direction * self.PLAYER_HOP_DISTANCE
            self.hop_cooldown = self.HOP_COOLDOWN_TICKS
            self._create_particles(self.player_pos + hop_direction * self.PLAYER_RADIUS, 10, self.COLOR_PARTICLE_THRUST, (10, 20), (1, 3), hop_direction)
            # SFX: Player Hop
            hop_executed = True

        if hop_executed:
             # Clamp target position to screen bounds
            self.player_target_pos.x = max(self.PLAYER_RADIUS, min(self.SCREEN_WIDTH - self.PLAYER_RADIUS, self.player_target_pos.x))
            self.player_target_pos.y = max(self.PLAYER_RADIUS, min(self.SCREEN_HEIGHT - self.PLAYER_RADIUS, self.player_target_pos.y))

        if shift_pressed and not self.shield_active and self.time_remaining_ticks > self.SHIELD_COST_TICKS:
            self.shield_active = True
            self.shield_timer = self.SHIELD_DURATION_TICKS
            self.time_remaining_ticks -= self.SHIELD_COST_TICKS
            # SFX: Shield Activate
            
    def _update_game_state(self):
        self.logic_ticks += 1
        self.stage_timer += 1
        self.time_remaining_ticks -= 1
        if self.hop_cooldown > 0: self.hop_cooldown -= 1
        
        # Smooth player movement
        self.player_pos.move_towards_ip(self.player_target_pos, 5)

        # Update shield
        if self.shield_active:
            self.shield_timer -= 1
            if self.shield_timer <= 0:
                self.shield_active = False
                # SFX: Shield Deactivate

        # Update obstacles
        for obs in self.obstacles:
            obs['pos'] += obs['vel']
            # Screen wrapping
            if obs['pos'].x < -obs['radius']: obs['pos'].x = self.SCREEN_WIDTH + obs['radius']
            if obs['pos'].x > self.SCREEN_WIDTH + obs['radius']: obs['pos'].x = -obs['radius']
            if obs['pos'].y < -obs['radius']: obs['pos'].y = self.SCREEN_HEIGHT + obs['radius']
            if obs['pos'].y > self.SCREEN_HEIGHT + obs['radius']: obs['pos'].y = -obs['radius']
            
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['start_radius'] * (p['life'] / p['start_life']))

    def _calculate_reward(self):
        reward = 0
        
        # Distance-to-goal reward
        goal_center = pygame.Vector2(self.goal_rect.center)
        current_dist = self.player_pos.distance_to(goal_center)
        reward += (self.last_dist_to_goal - current_dist) * 0.01 # Scaled down
        self.last_dist_to_goal = current_dist
        
        # Safe zone penalty
        if (self.player_pos.x < self.SAFE_ZONE_MARGIN or 
            self.player_pos.x > self.SCREEN_WIDTH - self.SAFE_ZONE_MARGIN or
            self.player_pos.y < self.SAFE_ZONE_MARGIN or
            self.player_pos.y > self.SCREEN_HEIGHT - self.SAFE_ZONE_MARGIN):
            reward -= 0.02
        
        return reward

    def _check_collisions(self):
        # Player vs Obstacles
        if not self.shield_active:
            for obs in self.obstacles:
                if self.player_pos.distance_to(obs['pos']) < self.PLAYER_RADIUS + obs['radius']:
                    self.game_over = True
                    self.score -= 10
                    self._create_particles(self.player_pos, 50, self.COLOR_PARTICLE_EXPLOSION, (20, 40), (2, 6))
                    # SFX: Player Explosion
                    return

        # Player vs Goal
        if self.goal_rect.collidepoint(self.player_pos):
            self.game_over = True
            self.score += 100
            # SFX: Goal Reached
            return

        # Player vs Checkpoint
        if not self.checkpoint_reached and self.checkpoint_rect.collidepoint(self.player_pos):
            self.checkpoint_reached = True
            self.score += 5
            # SFX: Checkpoint
            return

    def _update_stage(self):
        if self.stage < 3 and self.stage_timer >= self.STAGE_DURATION_TICKS:
            self.stage += 1
            self.stage_timer = 0
            self._setup_stage()
            # SFX: Stage Up

    def _setup_stage(self):
        self.obstacles.clear()
        
        stage_configs = {
            1: {'count': 15, 'speed': 1.0},
            2: {'count': 20, 'speed': 1.5},
            3: {'count': 25, 'speed': 2.0},
        }
        config = stage_configs[self.stage]
        
        for _ in range(config['count']):
            while True:
                pos = pygame.Vector2(
                    self.np_random.uniform(0, self.SCREEN_WIDTH),
                    self.np_random.uniform(0, self.SCREEN_HEIGHT)
                )
                # Avoid spawning on player
                if pos.distance_to(self.player_pos) > 100:
                    break
            
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5) * config['speed']
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            radius = self.np_random.integers(10, 20)
            
            self.obstacles.append({'pos': pos, 'vel': vel, 'radius': radius})

    def _setup_starfield(self):
        self.stars = []
        for i in range(3): # 3 layers of stars
            for _ in range(50):
                self.stars.append({
                    'pos': pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)),
                    'size': i + 1,
                    'speed': (i + 1) * 0.1
                })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Starfield
        for star in self.stars:
            star['pos'].x = (star['pos'].x - star['speed']) % self.SCREEN_WIDTH
            pygame.draw.circle(self.screen, (star['size']*40, star['size']*40, star['size']*50), star['pos'], star['size'])

        # Safe Zone visualization
        safe_zone_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        safe_zone_surf.fill((0,0,0,0))
        pygame.draw.rect(safe_zone_surf, self.COLOR_SAFE_ZONE + (30,), (0,0,self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.draw.rect(safe_zone_surf, (0,0,0,0), (self.SAFE_ZONE_MARGIN, self.SAFE_ZONE_MARGIN, self.SCREEN_WIDTH - 2*self.SAFE_ZONE_MARGIN, self.SCREEN_HEIGHT - 2*self.SAFE_ZONE_MARGIN))
        self.screen.blit(safe_zone_surf, (0,0))

        # Goal and Checkpoint
        goal_surf = pygame.Surface(self.goal_rect.size, pygame.SRCALPHA)
        goal_surf.fill(self.COLOR_GOAL + (50,))
        self.screen.blit(goal_surf, self.goal_rect.topleft)
        
        if not self.checkpoint_reached:
            checkpoint_surf = pygame.Surface(self.checkpoint_rect.size, pygame.SRCALPHA)
            checkpoint_surf.fill(self.COLOR_CHECKPOINT + (50,))
            self.screen.blit(checkpoint_surf, self.checkpoint_rect.topleft)

        # Particles
        for p in self.particles:
            if p['radius'] > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'] + (int(255 * p['life']/p['start_life']),) )
        
        # Obstacles
        for obs in self.obstacles:
            pos_int = (int(obs['pos'].x), int(obs['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], obs['radius'], self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], obs['radius'], self.COLOR_OBSTACLE_OUTLINE)
        
        # Player
        pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Shield effect
        if self.shield_active:
            shield_radius = self.PLAYER_RADIUS + 6 + 3 * math.sin(self.logic_ticks * 0.5)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(shield_radius), self.COLOR_PLAYER_SHIELD)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(shield_radius), (200,220,255,180))

        # Player ship (triangle)
        p1 = self.player_pos + pygame.Vector2(self.PLAYER_RADIUS, 0)
        p2 = self.player_pos + pygame.Vector2(-self.PLAYER_RADIUS/2, -self.PLAYER_RADIUS*0.866)
        p3 = self.player_pos + pygame.Vector2(-self.PLAYER_RADIUS/2, self.PLAYER_RADIUS*0.866)
        pygame.gfxdraw.aapolygon(self.screen, [p1,p2,p3], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1,p2,p3], self.COLOR_PLAYER)

    def _render_ui(self):
        # Time
        time_text = f"TIME: {self.time_remaining_ticks // self.LOGIC_TICK_RATE:03d}"
        time_surf = self.font_main.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (20, 10))

        # Score
        score_text = f"SCORE: {self.score:04d}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 20, 10))
        
        # Stage
        stage_text = f"STAGE: {self.stage}"
        stage_surf = self.font_small.render(stage_text, True, self.COLOR_TEXT)
        self.screen.blit(stage_surf, (self.SCREEN_WIDTH/2 - stage_surf.get_width()/2, 15))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "time_remaining": self.time_remaining_ticks,
            "checkpoint_reached": self.checkpoint_reached
        }

    def _create_particles(self, pos, count, color, life_range, speed_range, direction=None):
        for _ in range(count):
            if direction:
                angle = math.atan2(direction.y, direction.x) + self.np_random.uniform(-0.5, 0.5) - math.pi
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
            
            speed = self.np_random.uniform(*speed_range)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(*life_range)
            
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': life,
                'start_life': life,
                'color': color,
                'radius': self.np_random.integers(2, 5),
                'start_radius': 4
            })
            
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Hopper")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("="*30 + "\n")

    while running:
        movement = 0 # no-op
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
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Convert observation back to a surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Score: {info['score']}. Press 'R' to restart.")
            
        clock.tick(GameEnv.FPS)
        
    env.close()