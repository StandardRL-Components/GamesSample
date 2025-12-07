
# Generated: 2025-08-28T05:57:58.572995
# Source Brief: brief_02787.md
# Brief Index: 2787

        
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
        "Controls: Press space to jump over obstacles. Time your jumps with the beat to build your combo."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-futuristic rhythm runner. Jump over obstacles on the highway, timing your moves to the beat of the music to score big. Survive three stages to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 15, 35)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 200, 255)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_PARTICLE = (50, 255, 150)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_BEAT_INDICATOR = (255, 255, 255)
    COLOR_LANE_LINES = [(50, 50, 70), (40, 40, 60), (30, 30, 50)]

    # Game parameters
    FPS = 30
    BPM = 120
    BEAT_INTERVAL = (60 / BPM) * FPS
    BEAT_WINDOW = 3 # frames on either side of a beat

    MAX_LIVES = 3
    TOTAL_STAGES = 3
    STAGE_DURATION_SECONDS = 60
    MAX_STEPS = STAGE_DURATION_SECONDS * FPS * TOTAL_STAGES

    # Player physics
    PLAYER_X = 100
    PLAYER_BASE_SIZE = 20
    GROUND_Y = SCREEN_HEIGHT - 60
    GRAVITY = 0.8
    JUMP_STRENGTH = -14

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
        self.font_medium = pygame.font.Font(None, 48)
        self.font_large = pygame.font.Font(None, 72)
        
        # Etc...        
        self.obstacles = []
        self.particles = []
        self.lane_lines = []
        
        self.prev_space_held = False
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.lives = self.MAX_LIVES
        self.combo = 1
        
        self.player_y = self.GROUND_Y
        self.player_vy = 0
        self.is_jumping = False
        self.jump_was_on_beat = False
        
        self.obstacles = []
        self.particles = []
        
        self.beat_timer = 0
        self.obstacle_spawn_timer = self.FPS * 2 # Grace period

        self._init_lane_lines()
        
        self.stage_transition_effect = 0
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # --- Handle Input ---
        jump_triggered = space_held and not self.prev_space_held
        if jump_triggered and not self.is_jumping:
            self._player_jump()
        self.prev_space_held = space_held
        
        # --- Update Game Logic ---
        self.steps += 1
        events = self._update_state(jump_triggered)
        
        reward = self._calculate_reward(events)
        self.score += reward
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            terminal_reward = 0
            if self.lives <= 0:
                terminal_reward = -100 # Lose game penalty
            elif self.steps >= self.MAX_STEPS:
                terminal_reward = 300 # Win game bonus
            reward += terminal_reward
            self.score += terminal_reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _player_jump(self):
        self.is_jumping = True
        self.player_vy = self.JUMP_STRENGTH
        # sfx: player_jump.wav
        
        # Check if jump was on beat
        is_on_beat = abs(self.beat_timer) < self.BEAT_WINDOW or abs(self.beat_timer - self.BEAT_INTERVAL) < self.BEAT_WINDOW
        self.jump_was_on_beat = is_on_beat

    def _update_state(self, jump_triggered):
        events = {
            "survived_step": True,
            "jumped_off_beat": False,
            "cleared_obstacle_on_beat": 0,
            "collision": False,
            "stage_cleared": False
        }
        
        # Update timers
        self.beat_timer = (self.beat_timer + 1) % self.BEAT_INTERVAL
        if self.obstacle_spawn_timer > 0:
            self.obstacle_spawn_timer -= 1

        if jump_triggered and not self.is_jumping and not self.jump_was_on_beat:
            events["jumped_off_beat"] = True
            self.combo = 1
        
        # Update player physics
        self.player_vy += self.GRAVITY
        self.player_y += self.player_vy
        if self.player_y >= self.GROUND_Y:
            self.player_y = self.GROUND_Y
            self.player_vy = 0
            if self.is_jumping:
                self.is_jumping = False
                # sfx: player_land.wav

        # Update stage
        current_stage = self._get_current_stage()
        prev_stage = self._get_current_stage(steps_offset=-1)
        if current_stage > prev_stage:
            events["stage_cleared"] = True
            self.stage_transition_effect = self.FPS # 1 second effect
            # sfx: stage_clear.wav
        if self.stage_transition_effect > 0:
            self.stage_transition_effect -= 1

        # Update and spawn obstacles
        obstacle_speed = 8 * (1 + 0.1 * (current_stage - 1))
        self._update_obstacles(obstacle_speed, events)
        self._spawn_obstacles(current_stage)
        
        # Update particles and lane lines
        self._update_particles()
        self._update_lane_lines()
        
        return events
    
    def _calculate_reward(self, events):
        reward = 0.0
        if events["survived_step"]: reward += 0.1
        if events["jumped_off_beat"]: reward -= 0.2
        if events["cleared_obstacle_on_beat"] > 0:
            # sfx: success_ding.wav
            for _ in range(events["cleared_obstacle_on_beat"]):
                reward += 1.0
                reward += 2.0 * max(0, self.combo - 1)
                self.combo += 1
        if events["stage_cleared"]: reward += 100.0
        
        return reward

    def _check_termination(self):
        return self.lives <= 0 or self.steps >= self.MAX_STEPS

    def _get_current_stage(self, steps_offset=0):
        return min(self.TOTAL_STAGES, 1 + (self.steps + steps_offset) // (self.STAGE_DURATION_SECONDS * self.FPS))

    def _init_lane_lines(self):
        self.lane_lines = []
        for i in range(3):
            y = self.GROUND_Y + 15 + i * 20
            for j in range(self.SCREEN_WIDTH // 40 + 2):
                x = j * 40
                self.lane_lines.append({"x": x, "y": y, "layer": i})

    def _update_lane_lines(self):
        for line in self.lane_lines:
            line["x"] -= 2 * (3 - line["layer"]) # Parallax effect
            if line["x"] < -20:
                line["x"] += self.SCREEN_WIDTH + 40

    def _spawn_obstacles(self, stage):
        if self.obstacle_spawn_timer <= 0:
            # Only spawn on a beat for rhythmic patterns
            if self.beat_timer == 0:
                spawn_chance = 0.5 + 0.15 * (stage - 1)
                if self.np_random.random() < spawn_chance:
                    height = self.np_random.integers(30, 61)
                    self.obstacles.append({
                        "x": self.SCREEN_WIDTH + 10,
                        "w": 25,
                        "h": height,
                        "cleared": False
                    })
                    # sfx: obstacle_spawn.wav
                
                # Reset timer to a multiple of beat interval
                intervals = [3, 4, 5] if stage == 1 else [2, 3] if stage == 2 else [1, 2]
                self.obstacle_spawn_timer = self.np_random.choice(intervals) * self.BEAT_INTERVAL

    def _update_obstacles(self, speed, events):
        player_rect = pygame.Rect(self.PLAYER_X - self.PLAYER_BASE_SIZE // 2, self.player_y - self.PLAYER_BASE_SIZE, self.PLAYER_BASE_SIZE, self.PLAYER_BASE_SIZE)
        
        for obs in self.obstacles:
            obs["x"] -= speed
            
            obs_rect = pygame.Rect(obs["x"], self.GROUND_Y - obs["h"], obs["w"], obs["h"])
            
            # Collision check
            if not obs["cleared"] and player_rect.colliderect(obs_rect):
                obs["cleared"] = True # Mark as handled
                self.lives -= 1
                self.combo = 1
                events["collision"] = True
                # sfx: collision_hit.wav
                for _ in range(15):
                    self.particles.append(self._create_particle(self.PLAYER_X, self.player_y, self.COLOR_OBSTACLE))
            
            # Successful jump check
            if not obs["cleared"] and obs["x"] + obs["w"] < self.PLAYER_X:
                obs["cleared"] = True
                if self.is_jumping:
                    if self.jump_was_on_beat:
                        events["cleared_obstacle_on_beat"] += 1
                        for _ in range(20):
                            self.particles.append(self._create_particle(self.PLAYER_X, self.player_y, self.COLOR_PARTICLE))
                    else:
                        self.combo = 1 # Reset combo on off-beat clear
        
        self.obstacles = [obs for obs in self.obstacles if obs["x"] + obs["w"] > 0]
    
    def _create_particle(self, x, y, color):
        angle = self.np_random.random() * 2 * math.pi
        speed = self.np_random.random() * 3 + 1
        return {
            "x": x, "y": y,
            "vx": math.cos(angle) * speed,
            "vy": math.sin(angle) * speed,
            "life": self.np_random.integers(15, 30),
            "color": color
        }

    def _update_particles(self):
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

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
            "lives": self.lives,
            "combo": self.combo,
            "stage": self._get_current_stage(),
        }

    def _render_game(self):
        # Render lane lines
        for line in self.lane_lines:
            pygame.draw.rect(self.screen, self.COLOR_LANE_LINES[line["layer"]], (int(line["x"]), int(line["y"]), 20, 2))
        
        # Render ground
        pygame.draw.line(self.screen, self.COLOR_LANE_LINES[0], (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 2)
        
        # Render obstacles
        for obs in self.obstacles:
            rect = (int(obs["x"]), int(self.GROUND_Y - obs["h"]), int(obs["w"]), int(obs["h"]))
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)
            pygame.draw.rect(self.screen, (255,150,150), rect, 2) # Highlight

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30))
            color = p["color"]
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["x"]), int(p["y"]), 2, (*color, alpha))
            
        # Render player
        self._render_player()

        # Render beat indicator
        pulse = abs(math.sin((self.beat_timer / self.BEAT_INTERVAL) * math.pi))
        size = int(10 + 10 * pulse)
        alpha = int(50 + 205 * pulse)
        pygame.gfxdraw.filled_circle(self.screen, self.PLAYER_X, self.SCREEN_HEIGHT - 20, size, (*self.COLOR_BEAT_INDICATOR, alpha))
        pygame.gfxdraw.aacircle(self.screen, self.PLAYER_X, self.SCREEN_HEIGHT - 20, size, (*self.COLOR_BEAT_INDICATOR, alpha))

    def _render_player(self):
        size = self.PLAYER_BASE_SIZE
        y = self.player_y
        
        points = [
            (self.PLAYER_X, y - size),
            (self.PLAYER_X - size / 2, y),
            (self.PLAYER_X + size / 2, y),
        ]
        
        # Glow effect
        for i in range(5, 0, -1):
            alpha = 150 - i * 30
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], (*self.COLOR_PLAYER_GLOW, alpha))

        # Main player body
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_PLAYER_GLOW)


    def _render_ui(self):
        # Combo
        combo_text = self.font_medium.render(f"x{self.combo}", True, self.COLOR_UI_TEXT)
        self.screen.blit(combo_text, (20, 20))
        
        # Lives
        for i in range(self.lives):
            points = [
                (self.SCREEN_WIDTH - 30 - i * 25, 30),
                (self.SCREEN_WIDTH - 40 - i * 25, 40),
                (self.SCREEN_WIDTH - 20 - i * 25, 40)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        
        # Stage
        stage_text = self.font_small.render(f"STAGE {self._get_current_stage()}/{self.TOTAL_STAGES}", True, self.COLOR_UI_TEXT)
        text_rect = stage_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20))
        self.screen.blit(stage_text, text_rect)
        
        # Stage transition / Game Over text
        if self.stage_transition_effect > 0 and self.steps < self.MAX_STEPS - self.FPS:
            alpha = int(255 * min(1.0, (self.stage_transition_effect / (self.FPS/2))))
            stage_announce_text = self.font_large.render(f"STAGE {self._get_current_stage()}", True, (*self.COLOR_UI_TEXT, alpha))
            text_rect = stage_announce_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(stage_announce_text, text_rect)
        elif self.game_over:
            end_text_str = "GAME OVER" if self.lives <= 0 else "YOU WIN!"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the game directly to test it
    env = GameEnv()
    env.reset()
    
    running = True
    
    keys = {pygame.K_SPACE: 0}
    action = env.action_space.sample()
    action.fill(0)
    
    screen_for_display = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Rhythm Jumper")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in keys: keys[event.key] = 1
                if event.key == pygame.K_r: env.reset()
            if event.type == pygame.KEYUP:
                if event.key in keys: keys[event.key] = 0

        action[1] = keys[pygame.K_SPACE]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_for_display.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            env.reset()
            
        env.clock.tick(env.FPS)
        
    env.close()